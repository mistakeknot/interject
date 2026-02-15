"""Embedding client — delegates to intersearch shared library.

Re-exports all public API for backward compatibility.
"""

from intersearch.embeddings import (  # noqa: F401
    DEFAULT_MODEL,
    EMBEDDING_DIM,
    EmbeddingClient,
    bytes_to_vector,
    vector_to_bytes,
)
