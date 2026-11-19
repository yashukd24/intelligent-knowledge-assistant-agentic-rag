"""
backend/services/embedding_service.py
─────────────────────────────────────────────────────────────
Isolated Embedding Service using Sentence Transformers.

Features:
  - Model: BAAI/bge-small-en-v1.5 (default, configurable)
  - Execution: CPU-first with configurable batch size and device
  - Output: 384-dimensional unit-normalized float32 vectors
  - Thread-safe process singleton model caching with lazy loading
  - Pre-flight RAM safety guardrail (>=500 MB)
  - Strict input validation and custom error hierarchy
  - Optional dependency injection for zero-network testing
  - Similarity computation via cosine/dot product
─────────────────────────────────────────────────────────────
"""

import logging
import math
import os
import threading
from typing import Any, List, Optional, Sequence

from backend.config import Settings, get_settings
from backend.schemas.chunk import DocumentChunk
from backend.schemas.embedding import EmbeddedChunk

logger = logging.getLogger(__name__)


# ── Exception Hierarchy ──────────────────────────────────────────────────────


class EmbeddingServiceError(Exception):
    """Base exception for all embedding service failures."""
    pass


class ModelLoadError(EmbeddingServiceError):
    """Raised when the embedding model fails to load (OOM, missing files, corrupted cache)."""
    pass


class EmbeddingInferenceError(EmbeddingServiceError):
    """Raised when model inference fails during text encoding."""
    pass


class InvalidInputError(EmbeddingServiceError):
    """Raised when input text, sequence, or parameters fail validation."""
    pass


# ── Embedding Service ────────────────────────────────────────────────────────


class EmbeddingService:
    """
    Isolated embedding service for converting text chunks into dense vector representations.
    """

    _cached_model: Optional[Any] = None
    _model_lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        model: Optional[Any] = None,
        settings: Optional[Settings] = None,
        model_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        normalize: Optional[bool] = None,
        min_ram_mb: int = 500,
    ):
        """
        Initialize the embedding service.

        Args:
            model: Optional pre-loaded or mock model for dependency injection.
            settings: Optional application settings. Falls back to get_settings().
            model_name: Override model identifier (e.g. 'BAAI/bge-small-en-v1.5').
            cache_dir: Override Hugging Face model cache directory.
            device: Override inference device ('cpu', 'cuda').
            batch_size: Override encoding batch size.
            normalize: Whether to L2-normalize embeddings (default True).
            min_ram_mb: Minimum available RAM in MB required to load the model (default 500 MB).
        """
        self.settings = settings or get_settings()
        self.model_name = model_name or self.settings.embedding_model
        self.cache_dir = cache_dir or self.settings.effective_embedding_cache_dir
        self.device = device or self.settings.embedding_device
        self.batch_size = batch_size or self.settings.embedding_batch_size
        self.normalize = normalize if normalize is not None else self.settings.embedding_normalize
        self.min_ram_mb = min_ram_mb

        self._injected_model = model
        self._dimension: Optional[int] = None

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the process-level cached model instance."""
        with cls._model_lock:
            cls._cached_model = None
            logger.info("EmbeddingService process model cache cleared.")

    def _check_ram_safety(self) -> None:
        """
        Check that sufficient system RAM is available before loading the model.
        Raises ModelLoadError if available RAM is below the safety threshold.
        """
        try:
            import gc
            import psutil

            available_mb = psutil.virtual_memory().available / (1024 * 1024)
            logger.info(
                "Pre-flight RAM check: %.1f MB available (threshold: %d MB).",
                available_mb,
                self.min_ram_mb,
            )
            if available_mb < self.min_ram_mb:
                # Attempt garbage collection to reclaim memory held by cyclic references
                gc.collect()
                available_mb = psutil.virtual_memory().available / (1024 * 1024)
                logger.info(
                    "Post-GC RAM check: %.1f MB available (threshold: %d MB).",
                    available_mb,
                    self.min_ram_mb,
                )
                if available_mb < self.min_ram_mb:
                    raise ModelLoadError(
                        f"Insufficient RAM to load embedding model '{self.model_name}': "
                        f"{available_mb:.1f} MB available, minimum required is {self.min_ram_mb} MB."
                    )
        except ImportError:
            logger.warning("psutil not available; skipping pre-flight RAM safety check.")

    def _get_model(self) -> Any:
        """
        Retrieve or lazy-load the SentenceTransformer model instance.
        Uses process-level singleton caching with thread-safe double-checked locking.
        """
        if self._injected_model is not None:
            return self._injected_model

        if EmbeddingService._cached_model is not None:
            return EmbeddingService._cached_model

        with EmbeddingService._model_lock:
            if EmbeddingService._cached_model is not None:
                return EmbeddingService._cached_model

            # 1. Check RAM safety threshold
            self._check_ram_safety()

            # 2. Configure Hugging Face cache directory if specified
            if self.cache_dir:
                os.environ["HF_HOME"] = self.cache_dir
                logger.info("Hugging Face cache directory set to '%s'.", self.cache_dir)

            # 3. Load SentenceTransformer model
            logger.info(
                "Loading SentenceTransformer model '%s' on device '%s'...",
                self.model_name,
                self.device,
            )
            try:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    cache_folder=self.cache_dir,
                    trust_remote_code=False,
                )
                EmbeddingService._cached_model = model
                logger.info("Model '%s' successfully loaded into process cache.", self.model_name)
                return model
            except Exception as exc:
                logger.error("Failed to load model '%s': %s", self.model_name, exc, exc_info=True)
                raise ModelLoadError(
                    f"Failed to load embedding model '{self.model_name}': {exc}"
                ) from exc

    @property
    def dimension(self) -> int:
        """
        Return the embedding vector dimension verified from the loaded model
        or configuration.
        """
        if self._dimension is not None:
            return self._dimension

        model = self._get_model()
        if hasattr(model, "get_sentence_embedding_dimension"):
            dim = model.get_sentence_embedding_dimension()
            if dim:
                self._dimension = int(dim)
                return self._dimension

        # Fallback to configuration
        self._dimension = self.settings.embedding_dimension
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        """
        Generate a unit-normalized embedding vector for a single text string.

        Args:
            text: Non-empty, non-whitespace string to embed.

        Returns:
            List of floats representing the dense embedding vector.

        Raises:
            InvalidInputError: If text is empty, whitespace, or not a string.
            EmbeddingInferenceError: If model inference fails.
        """
        if not isinstance(text, str):
            raise InvalidInputError(f"Input text must be a string, got {type(text).__name__}")
        if not text.strip():
            raise InvalidInputError("Input text cannot be empty or whitespace only")

        embeddings = self.embed_batch([text], batch_size=1)
        return embeddings[0]

    def embed_batch(
        self,
        texts: Sequence[str],
        batch_size: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Generate unit-normalized embedding vectors for a sequence of texts.

        Args:
            texts: Non-empty sequence of non-empty, non-whitespace strings.
            batch_size: Optional batch size override. Defaults to configured batch size.

        Returns:
            List of embedding vectors, each vector being a List[float].

        Raises:
            InvalidInputError: If texts is empty, not a sequence, or contains invalid strings.
            EmbeddingInferenceError: If model inference fails.
        """
        if texts is None or not isinstance(texts, Sequence) or isinstance(texts, (str, bytes)):
            raise InvalidInputError("Input texts must be a non-string sequence of strings")
        if len(texts) == 0:
            raise InvalidInputError("Input texts sequence cannot be empty")

        for i, t in enumerate(texts):
            if not isinstance(t, str):
                raise InvalidInputError(f"Text at index {i} must be a string, got {type(t).__name__}")
            if not t.strip():
                raise InvalidInputError(f"Text at index {i} cannot be empty or whitespace only")

        eff_batch_size = self.batch_size if batch_size is None else batch_size
        if eff_batch_size < 1:
            raise InvalidInputError(f"Batch size must be >= 1, got {eff_batch_size}")

        model = self._get_model()

        try:
            raw_embeddings = model.encode(
                list(texts),
                batch_size=eff_batch_size,
                show_progress_bar=False,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
            )
        except Exception as exc:
            logger.error("Embedding inference error: %s", exc, exc_info=True)
            raise EmbeddingInferenceError(f"Embedding inference failed: {exc}") from exc

        # Convert output to standard Python List[List[float]]
        result: List[List[float]] = []
        for vec in raw_embeddings:
            if hasattr(vec, "tolist"):
                vector_list = [float(x) for x in vec.tolist()]
            else:
                vector_list = [float(x) for x in vec]
            result.append(vector_list)

        return result

    def embed_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        batch_size: Optional[int] = None,
    ) -> List[EmbeddedChunk]:
        """
        Generate normalized embeddings for a sequence of DocumentChunk objects
        and return corresponding EmbeddedChunk instances with preserved provenance.

        Args:
            chunks: Sequence of DocumentChunk objects.
            batch_size: Optional batch size override.

        Returns:
            List of EmbeddedChunk instances with dense vectors and metadata.

        Raises:
            InvalidInputError: If chunks is not a sequence or contains invalid elements.
            EmbeddingInferenceError: If model inference fails.
        """
        if chunks is None or not isinstance(chunks, Sequence) or isinstance(chunks, (str, bytes)):
            raise InvalidInputError("Input chunks must be a sequence of DocumentChunk objects")

        if len(chunks) == 0:
            return []

        texts: List[str] = []
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, DocumentChunk):
                raise InvalidInputError(
                    f"Item at index {i} is not a DocumentChunk instance, got {type(chunk).__name__}"
                )
            texts.append(chunk.chunk_text)

        vectors = self.embed_batch(texts, batch_size=batch_size)
        dim = self.dimension

        embedded_chunks: List[EmbeddedChunk] = []
        for chunk, vec in zip(chunks, vectors):
            ec = EmbeddedChunk.from_document_chunk(
                chunk=chunk,
                embedding=vec,
                embedding_dimension=dim,
                model_name=self.model_name,
            )
            embedded_chunks.append(ec)

        return embedded_chunks

    @staticmethod
    def compute_similarity(
        embedding_a: Sequence[float],
        embedding_b: Sequence[float],
    ) -> float:
        """
        Compute cosine similarity between two embedding vectors.

        For unit-normalized vectors (L2 norm = 1.0), cosine similarity is identical
        to the dot product:
            cos(a, b) = dot(a, b) / (||a|| * ||b||) = dot(a, b)
        This method computes full cosine similarity to guarantee accuracy even
        if vectors are not pre-normalized.

        Args:
            embedding_a: First embedding vector.
            embedding_b: Second embedding vector.

        Returns:
            Cosine similarity score as float in range [-1.0, 1.0].

        Raises:
            InvalidInputError: If vectors have mismatched dimensions, are empty, or contain non-finite numbers.
        """
        if embedding_a is None or embedding_b is None:
            raise InvalidInputError("Embeddings cannot be None")
        if len(embedding_a) == 0 or len(embedding_b) == 0:
            raise InvalidInputError("Embeddings cannot be empty")
        if len(embedding_a) != len(embedding_b):
            raise InvalidInputError(
                f"Vector dimensions do not match: {len(embedding_a)} vs {len(embedding_b)}"
            )

        dot_product = 0.0
        norm_a_sq = 0.0
        norm_b_sq = 0.0

        for idx, (a, b) in enumerate(zip(embedding_a, embedding_b)):
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                raise InvalidInputError(f"Vector value at index {idx} must be numeric")
            if math.isnan(a) or math.isinf(a) or math.isnan(b) or math.isinf(b):
                raise InvalidInputError(f"Vector value at index {idx} must be finite (not NaN or Inf)")

            dot_product += float(a) * float(b)
            norm_a_sq += float(a) * float(a)
            norm_b_sq += float(b) * float(b)

        norm_a = math.sqrt(norm_a_sq)
        norm_b = math.sqrt(norm_b_sq)

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        similarity = dot_product / (norm_a * norm_b)
        # Clamp to [-1.0, 1.0] to guard against floating-point precision inaccuracies
        return max(-1.0, min(1.0, similarity))
