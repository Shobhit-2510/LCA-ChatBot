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

COLLECTION = "lca_book"  # Chroma collection name
FETCH_K = 30  # MMR fetches 30 candidates, re-ranks to TOP_K


@lru_cache(maxsize=1)
def get_store() -> Chroma:
    """Open the persisted Chroma collection built in Phase A."""
    return Chroma(
        collection_name=COLLECTION,  # Name of the collection
        embedding_function=get_embeddings(),  # Use same embeddings as indexing
        persist_directory=str(config.CHROMA_DIR),  # Load from saved directory
    )


def get_retriever():
    """Return an MMR retriever with the paper's settings."""
    return get_store().as_retriever(
        search_type=config.SEARCH_TYPE,  # Maximum Marginal Relevance
        search_kwargs={
            "k": config.TOP_K,  # Return 10 chunks
            "fetch_k": FETCH_K,  # Fetch 30 candidates, then pick the 10 best (diverse + relevant)
            "lambda_mult": config.MMR_LAMBDA,  # Balance relevance and diversity
        },
    )
