"""Step 4 — embeddings provider.

Wraps the sentence-transformers model (EMBEDDING_MODEL from config) so the
SAME model is used for both indexing (Phase A) and querying (Phase B).

BGE specifics:
  * embeddings are L2-normalized so cosine similarity == dot product;
  * queries get the recommended retrieval instruction prefix, documents
    do not — this is what BGE was trained for and it lifts recall.

langchain-huggingface's HuggingFaceEmbeddings has no `query_instruction`
arg, so a thin subclass prepends the BGE prefix in embed_query().

Public API:
    get_embeddings() -> Embeddings
"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

import config

# BGE's recommended retrieval instruction, prepended to queries only.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class BGEEmbeddings(HuggingFaceEmbeddings):
    """HuggingFaceEmbeddings that prepends BGE's query instruction."""

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(_BGE_QUERY_INSTRUCTION + text)


@lru_cache(maxsize=1)
def get_embeddings() -> BGEEmbeddings:
    """Return a cached bge-large-en-v1.5 embeddings client (CPU)."""
    return BGEEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},        # documents
        query_encode_kwargs={"normalize_embeddings": True},  # queries (cosine)
    )
