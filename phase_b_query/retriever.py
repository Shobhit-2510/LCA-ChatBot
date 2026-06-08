"""Load the Chroma store and expose an MMR retriever.

Uses SEARCH_TYPE="mmr", TOP_K=10, MMR_LAMBDA from config so retrieved
chunks are relevant AND non-redundant (paper's Eq. 1). The embedding
function must match the one used to build the index (bge-large-en-v1.5).

Public API:
    get_retriever() -> VectorStoreRetriever
    get_store() -> Chroma
"""

from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma

import config
from phase_a_ingestion.embed import get_embeddings

COLLECTION = "lca_book"
# MMR re-ranks the top FETCH_K candidates down to TOP_K; must exceed TOP_K.
FETCH_K = 30


@lru_cache(maxsize=1)
def get_store() -> Chroma:
    """Open the persisted Chroma collection built in Phase A."""
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )


def get_retriever():
    """Return an MMR retriever with the paper's settings."""
    return get_store().as_retriever(
        search_type=config.SEARCH_TYPE,  # "mmr"
        search_kwargs={
            "k": config.TOP_K,           # 10 chunks returned
            "fetch_k": FETCH_K,          # candidate pool MMR re-ranks
            "lambda_mult": config.MMR_LAMBDA,  # relevance vs. diversity
        },
    )
