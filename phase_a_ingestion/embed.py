"""Step 4 — embeddings provider.

Wraps the sentence-transformers model (EMBEDDING_MODEL from config) so the
SAME model is used for both indexing (Phase A) and querying (Phase B).

TODO: implement get_embeddings() -> Embeddings
"""
