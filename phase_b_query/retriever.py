"""Load the Chroma store and expose an MMR retriever.

Uses SEARCH_TYPE="mmr", TOP_K=10, MMR_LAMBDA from config so retrieved
chunks are relevant AND non-redundant (paper's Eq. 1).

TODO: implement get_retriever()
"""
