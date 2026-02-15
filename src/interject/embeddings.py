"""Embedding client with tldr-swinton sharing.

Primary path: call tldr-swinton's semantic MCP tool for embeddings.
Fallback: load sentence-transformers model locally.
"""

from __future__ import annotations

import logging
import struct
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default model — same as tldr-swinton uses for FAISS
DEFAULT_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension


class EmbeddingClient:
    """Provides text -> vector embeddings, sharing model with tldr-swinton when possible."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._local_model = None
        self._use_local = False

    def _ensure_local_model(self) -> None:
        """Lazy-load sentence-transformers model."""
        if self._local_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading local embedding model: %s", self.model_name)
            self._local_model = SentenceTransformer(self.model_name)
            self._use_local = True
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Install with: uv pip install sentence-transformers"
            )

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string. Returns a normalized vector."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts. Returns (N, dim) array of normalized vectors."""
        self._ensure_local_model()
        embeddings = self._local_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return np.array(embeddings, dtype=np.float32)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two normalized vectors."""
        return float(np.dot(a, b))


def vector_to_bytes(vec: np.ndarray) -> bytes:
    """Serialize a numpy vector to bytes for SQLite storage."""
    return vec.astype(np.float32).tobytes()


def bytes_to_vector(data: bytes) -> np.ndarray:
    """Deserialize bytes from SQLite back to numpy vector."""
    return np.frombuffer(data, dtype=np.float32)
