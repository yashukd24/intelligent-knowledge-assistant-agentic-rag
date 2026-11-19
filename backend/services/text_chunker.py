"""
backend/services/text_chunker.py
─────────────────────────────────────────────────────────────
Deterministic Semantic Text Chunking Service.

Key responsibilities:
  - Sliding-window character-based text chunking with configurable size and overlap
  - Word-boundary preservation (splits on sentences, paragraphs, or spaces when practical)
  - Accurate PDF page number and page range attribution
  - Assigns None for unpaginated formats (DOCX, TXT)
  - Zero duplicate, empty, or whitespace-only chunks
  - Guaranteed loop termination on unbroken tokens or edge cases
─────────────────────────────────────────────────────────────
"""

from typing import List, Optional, Tuple
from backend.config import get_settings
from backend.schemas.chunk import ChunkMetadata, DocumentChunk, TextPage
from backend.services.text_cleaner import TextCleaner


class TextChunker:
    """
    Deterministic sliding-window text chunker.
    
    IMPORTANT ARCHITECTURAL NOTE:
    Chunk sizing is character-based (default 512 chars with 64 chars overlap).
    Character-based sizing does NOT guarantee an exact LLM token limit.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        settings = get_settings()
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than chunk_size ({self.chunk_size})"
            )

    def _find_boundary(self, text: str, start: int, candidate_end: int) -> int:
        """
        Search backwards from candidate_end for a natural sentence, paragraph,
        or word boundary within a 20% tolerance window.
        """
        text_len = len(text)
        if candidate_end >= text_len:
            return text_len

        window_slice = text[start:candidate_end]
        tolerance = max(16, int(self.chunk_size * 0.20))
        min_split_point = max(start + 1, candidate_end - tolerance)

        # 1. Paragraph boundary (\n\n)
        last_para = text.rfind("\n\n", min_split_point, candidate_end)
        if last_para != -1:
            return last_para + 2

        # 2. Sentence boundary (. , ! , ? )
        for punct in [". ", "! ", "? "]:
            last_punct = text.rfind(punct, min_split_point, candidate_end)
            if last_punct != -1:
                return last_punct + len(punct)

        # 3. Line break (\n)
        last_nl = text.rfind("\n", min_split_point, candidate_end)
        if last_nl != -1:
            return last_nl + 1

        # 4. Word boundary (space)
        last_space = text.rfind(" ", min_split_point, candidate_end)
        if last_space != -1:
            return last_space + 1

        # Fallback: hard character cut if no delimiter exists within tolerance
        return candidate_end

    def _map_pages(
        self,
        start_char: int,
        end_char: int,
        page_offsets: List[Tuple[int, int, Optional[int]]],
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        Map character offsets in the combined document back to PDF page numbers.
        Returns (page_number, page_range).
        For unpaginated formats where all page_offsets have None, returns (None, None).
        """
        overlapping_pages: List[int] = []

        for p_start, p_end, p_num in page_offsets:
            if p_num is None:
                continue
            # Check overlap: max(start1, start2) < min(end1, end2)
            if max(start_char, p_start) < min(end_char, p_end):
                overlapping_pages.append(p_num)

        if not overlapping_pages:
            return None, None

        overlapping_pages = sorted(list(set(overlapping_pages)))
        if len(overlapping_pages) == 1:
            return overlapping_pages[0], None
        else:
            return overlapping_pages[0], f"{overlapping_pages[0]}-{overlapping_pages[-1]}"

    def chunk_document(
        self,
        document_id: str,
        collection_id: str,
        filename: str,
        file_type: str,
        pages: List[TextPage],
        version: Optional[int] = None,
    ) -> List[DocumentChunk]:
        """
        Chunk extracted document pages into structured DocumentChunk objects.

        Args:
            document_id: UUID of document.
            collection_id: UUID of collection.
            filename: Original filename.
            file_type: pdf, docx, or txt.
            pages: List of extracted TextPage objects.
            version: Optional document version for versioned chunk identification.

        Returns:
            Ordered list of DocumentChunk objects with accurate metadata.
        """
        if not pages:
            return []

        # 1. Clean individual pages and build continuous document with offset tracking
        combined_text_parts: List[str] = []
        page_offsets: List[Tuple[int, int, Optional[int]]] = []
        current_offset = 0

        for page in pages:
            cleaned_page = TextCleaner.clean(page.text)
            if not cleaned_page:
                continue

            page_len = len(cleaned_page)
            p_start = current_offset
            p_end = current_offset + page_len

            combined_text_parts.append(cleaned_page)
            page_offsets.append((p_start, p_end, page.page_number))

            current_offset = p_end + 2  # Separated by \n\n

        full_text = "\n\n".join(combined_text_parts).strip()
        if not full_text:
            return []

        total_len = len(full_text)
        id_prefix = f"{document_id}_v{version}" if version is not None else document_id

        # 2. If entire document is smaller than or equal to chunk_size, return 1 chunk
        if total_len <= self.chunk_size:
            page_num, page_rng = self._map_pages(0, total_len, page_offsets)
            meta = ChunkMetadata(
                document_id=document_id,
                collection_id=collection_id,
                filename=filename,
                file_type=file_type,
                chunk_index=0,
                char_count=total_len,
                page_number=page_num,
                page_range=page_rng,
                start_char=0,
                end_char=total_len,
                version=version,
            )
            return [
                DocumentChunk(
                    chunk_id=f"{id_prefix}_c0000",
                    document_id=document_id,
                    collection_id=collection_id,
                    filename=filename,
                    file_type=file_type,
                    chunk_index=0,
                    chunk_text=full_text,
                    char_count=total_len,
                    page_number=page_num,
                    page_range=page_rng,
                    metadata=meta,
                    version=version,
                )
            ]

        # 3. Sliding window chunking
        chunks: List[DocumentChunk] = []
        start_pos = 0
        chunk_idx = 0

        while start_pos < total_len:
            candidate_end = min(start_pos + self.chunk_size, total_len)
            split_pos = self._find_boundary(full_text, start_pos, candidate_end)

            chunk_raw = full_text[start_pos:split_pos]
            chunk_text = chunk_raw.strip()

            if chunk_text:
                # Calculate trimmed character offsets
                offset_delta_start = chunk_raw.find(chunk_text)
                actual_start = start_pos + max(0, offset_delta_start)
                actual_end = actual_start + len(chunk_text)

                page_num, page_rng = self._map_pages(actual_start, actual_end, page_offsets)

                meta = ChunkMetadata(
                    document_id=document_id,
                    collection_id=collection_id,
                    filename=filename,
                    file_type=file_type,
                    chunk_index=chunk_idx,
                    char_count=len(chunk_text),
                    page_number=page_num,
                    page_range=page_rng,
                    start_char=actual_start,
                    end_char=actual_end,
                    version=version,
                )

                chunk = DocumentChunk(
                    chunk_id=f"{id_prefix}_c{chunk_idx:04d}",
                    document_id=document_id,
                    collection_id=collection_id,
                    filename=filename,
                    file_type=file_type,
                    chunk_index=chunk_idx,
                    chunk_text=chunk_text,
                    char_count=len(chunk_text),
                    page_number=page_num,
                    page_range=page_rng,
                    metadata=meta,
                    version=version,
                )
                chunks.append(chunk)
                chunk_idx += 1

            # Advance window
            if split_pos >= total_len:
                break

            next_start = split_pos - self.chunk_overlap
            # Guard against infinite loops or non-progressing window
            if next_start <= start_pos:
                next_start = split_pos

            start_pos = next_start

        return chunks
