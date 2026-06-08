"""Step 5 — build the Chroma vector store (entry point for Phase A).

Ties the pipeline together: extract -> clean -> chunk -> embed -> persist
to CHROMA_DIR. Run once offline; rerun to rebuild the knowledge base.

Usage (once implemented):  python -m phase_a_ingestion.build_index

TODO: implement main()
"""
