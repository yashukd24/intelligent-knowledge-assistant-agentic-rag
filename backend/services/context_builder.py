"""
backend/services/context_builder.py
─────────────────────────────────────────────────────────────
ContextBuilder: Converts candidate RetrievedChunk objects into
a structured, source-grounded context block with deterministic
source citations ([S1], [S2], ...), metadata preservation,
and strict context character budget enforcement.

Character Budget vs. Tokenizer Limitations:
  A character budget is used here to enforce deterministic string length
  and memory bounds without introducing external LLM tokenizer dependencies
  (such as tiktoken or anthropic-tokenizer) into Phase 8.
  Note that English text typically averages ~4 characters per token,
  so a 12,000-character budget roughly corresponds to ~3,000 tokens.
  Character budgets do not account for subword token expansion in multilingual
  or code text; Phase 9 may introduce model-specific token counting.
─────────────────────────────────────────────────────────────
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from backend.schemas.rag import (
    DEFAULT_CONTEXT_BUDGET,
    MAX_CONTEXT_BUDGET,
    MIN_CONTEXT_BUDGET,
    ExcludedChunkInfo,
    SourceReference,
)
from backend.schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


class ContextBuildResult(BaseModel):
    """
    Result of context assembly, including formatted context text,
    structured source references, excluded chunks, and metadata.
    """
    assembled_context: str = Field(default="", description="Complete assembled context block.")
    sources: List[SourceReference] = Field(default_factory=list, description="Ordered source references.")
    excluded_chunks: List[ExcludedChunkInfo] = Field(default_factory=list, description="Chunks excluded from context.")
    context_character_count: int = Field(default=0, ge=0, description="Length of assembled context string.")
    source_metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary keyed by source_id.")
    has_evidence: bool = Field(default=False, description="True if at least one source was included.")

    model_config = ConfigDict(from_attributes=True)


class ContextBuilder:
    """
    Assembles structured, grounded context from retrieved semantic chunks.
    """

    def __init__(
        self,
        default_max_context_chars: int = DEFAULT_CONTEXT_BUDGET,
    ):
        """
        Initialize ContextBuilder with default context budget.

        Args:
            default_max_context_chars: Default maximum character budget.
        """
        if default_max_context_chars < MIN_CONTEXT_BUDGET or default_max_context_chars > MAX_CONTEXT_BUDGET:
            raise ValueError(
                f"default_max_context_chars must be between {MIN_CONTEXT_BUDGET} and {MAX_CONTEXT_BUDGET}, "
                f"got {default_max_context_chars}"
            )
        self.default_max_context_chars = default_max_context_chars

    def build_context(
        self,
        chunks: Optional[List[RetrievedChunk]],
        max_context_chars: Optional[int] = None,
        expected_collection_id: Optional[str] = None,
    ) -> ContextBuildResult:
        """
        Convert retrieved chunks into a deterministic, source-cited context block.

        Rules:
          1. Preserves retrieval ranking order (Rank 1 -> [S1], Rank 2 -> [S2], ...).
          2. Preserves exact chunk text without modification or hallucination.
          3. Assigns deterministic source IDs: [S1], [S2], [S3], ...
          4. Includes filename, document ID, chunk ID, page number/range, and similarity score.
          5. Excludes cross-collection chunks if expected_collection_id is supplied.
          6. Deduplicates chunks by chunk_id to avoid redundant content.
          7. Enforces whole-chunk inclusion under the character budget; chunks that
             do not fit are recorded in excluded_chunks with reason 'budget_exceeded'.
          8. Handles empty chunk lists gracefully by returning an empty context block
             and has_evidence=False.

        Args:
            chunks: List of RetrievedChunk objects in ranking order.
            max_context_chars: Optional override for maximum character budget.
            expected_collection_id: Optional collection UUID to enforce isolation.

        Returns:
            ContextBuildResult containing assembled text, sources, and exclusions.
        """
        budget = max_context_chars if max_context_chars is not None else self.default_max_context_chars
        if budget < MIN_CONTEXT_BUDGET or budget > MAX_CONTEXT_BUDGET:
            raise ValueError(
                f"max_context_chars must be between {MIN_CONTEXT_BUDGET} and {MAX_CONTEXT_BUDGET}, got {budget}"
            )

        # 1. Handle empty input
        if not chunks:
            return ContextBuildResult(
                assembled_context="",
                sources=[],
                excluded_chunks=[],
                context_character_count=0,
                source_metadata={},
                has_evidence=False,
            )

        seen_chunk_ids: set[str] = set()
        sources: List[SourceReference] = []
        excluded_chunks: List[ExcludedChunkInfo] = []
        context_blocks: List[str] = []
        source_metadata: Dict[str, Any] = {}
        current_char_count = 0
        source_idx = 1

        for chunk in chunks:
            chunk_text_len = len(chunk.text)

            # 2. Enforce collection boundary isolation
            if expected_collection_id is not None and str(chunk.collection_id) != str(expected_collection_id):
                logger.warning(
                    "Cross-collection chunk detected: chunk %s belongs to %s, expected %s",
                    chunk.chunk_id,
                    chunk.collection_id,
                    expected_collection_id,
                )
                excluded_chunks.append(
                    ExcludedChunkInfo(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        reason="cross_collection_mismatch",
                        char_count=chunk_text_len,
                    )
                )
                continue

            # 3. Deduplicate by chunk_id
            if chunk.chunk_id in seen_chunk_ids:
                logger.debug("Duplicate chunk ID skipped: %s", chunk.chunk_id)
                excluded_chunks.append(
                    ExcludedChunkInfo(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        reason="duplicate_chunk_id",
                        char_count=chunk_text_len,
                    )
                )
                continue

            # 4. Format page reference
            page_str = "Page N/A"
            if chunk.page_number is not None:
                if chunk.page_range and chunk.page_range != str(chunk.page_number):
                    page_str = f"Pages {chunk.page_range}"
                else:
                    page_str = f"Page {chunk.page_number}"
            elif chunk.page_range:
                page_str = f"Pages {chunk.page_range}"

            source_id = f"[S{source_idx}]"
            filename_str = chunk.filename or "Unknown file"

            # 5. Build source header and block
            header_lines = [
                f"{source_id} Source: {filename_str} (Doc ID: {chunk.document_id}, Chunk ID: {chunk.chunk_id}, {page_str})",
                f"Similarity Score: {chunk.similarity_score:.4f} (retrieval metric; not proof of factual correctness)",
                "---",
            ]
            header_text = "\n".join(header_lines)
            block_text = f"{header_text}\n{chunk.text}"

            # Calculate additional length including separator between blocks
            separator_len = 2 if context_blocks else 0  # "\n\n" between blocks
            block_total_len = len(block_text) + separator_len

            # 6. Whole-chunk budget enforcement
            if current_char_count + block_total_len > budget:
                logger.info(
                    "Context character budget reached (%d + %d > %d). Chunk %s excluded.",
                    current_char_count,
                    block_total_len,
                    budget,
                    chunk.chunk_id,
                )
                excluded_chunks.append(
                    ExcludedChunkInfo(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        reason="budget_exceeded",
                        char_count=chunk_text_len,
                    )
                )
                continue

            # 7. Append accepted chunk to context
            seen_chunk_ids.add(chunk.chunk_id)
            context_blocks.append(block_text)
            current_char_count += block_total_len

            # 8. Record structured source reference and metadata
            source_ref = SourceReference(
                source_id=source_id,
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                filename=chunk.filename,
                page_number=chunk.page_number,
                page_range=chunk.page_range,
                chunk_index=chunk.chunk_index,
                similarity_score=round(chunk.similarity_score, 4),
                char_count=chunk_text_len,
            )
            sources.append(source_ref)

            source_metadata[source_id] = {
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id,
                "filename": chunk.filename,
                "page_number": chunk.page_number,
                "page_range": chunk.page_range,
                "chunk_index": chunk.chunk_index,
                "similarity_score": round(chunk.similarity_score, 4),
                "distance": round(chunk.distance, 4),
                "version": chunk.version,
                "file_type": chunk.file_type,
            }
            source_idx += 1

        assembled_context = "\n\n".join(context_blocks)
        total_len = len(assembled_context)

        return ContextBuildResult(
            assembled_context=assembled_context,
            sources=sources,
            excluded_chunks=excluded_chunks,
            context_character_count=total_len,
            source_metadata=source_metadata,
            has_evidence=len(sources) > 0,
        )
