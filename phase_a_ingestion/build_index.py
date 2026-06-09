"""Phase A driver — extract -> clean -> chunk -> embed -> Chroma.

Runs the full ingestion pipeline: writes the tagged chunks to
data/processed/chunks.jsonl, embeds them with bge-large-en-v1.5, and
persists the vectors to a cosine Chroma collection at config.CHROMA_DIR.

Usage:  python -m phase_a_ingestion.build_index
"""

from __future__ import annotations

import json
import shutil
import sys

from langchain_chroma import Chroma

import config
from phase_a_ingestion.extract import extract
from phase_a_ingestion.clean import clean_pages
from phase_a_ingestion.chunk import chunk_pages
from phase_a_ingestion.embed import get_embeddings

PDF_PATH = config.DATA_RAW / "Textbook.pdf"  # Input PDF path
CHUNKS_OUT = config.DATA_PROCESSED / "chunks.jsonl"  # Output chunks file
COLLECTION = "lca_book"  # Chroma collection name
EMBED_BATCH = 256  # Batch size for embedding (memory/speed trade-off)


def run_ingestion() -> list:  # Extract, clean, and chunk the PDF
    if not PDF_PATH.exists():  # Check if PDF exists
        sys.exit(f"PDF not found: {PDF_PATH}")

    print(f"[1/3] Extracting text from {PDF_PATH.name} ...")
    pages, chapter_index = extract(str(PDF_PATH))  # Extract pages and chapter index from PDF
    print(f"      {len(pages)} pages, {len(chapter_index)} top-level TOC entries")

    print("[2/3] Cleaning headers/footers, hyphenation, ligatures ...")
    cleaned = clean_pages(pages, chapter_index)  # Remove headers, footers, fix formatting
    print(f"      {len(cleaned)} non-empty pages after cleaning")

    print(f"[3/3] Chunking (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}) ...")
    docs = chunk_pages(cleaned, chapter_index)  # Split into overlapping chunks with metadata
    print(f"      {len(docs)} chunks produced")

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)  # Create output directory
    with open(CHUNKS_OUT, "w", encoding="utf-8") as f:  # Write chunks to JSONL
        for d in docs:
            f.write(json.dumps({"text": d.page_content, "metadata": d.metadata}) + "\n")
    print(f"      saved -> {CHUNKS_OUT}")

    return docs


def build_vector_store(docs: list) -> Chroma:  # Embed and persist chunks to vector store
    """Embed chunks with bge-large-en-v1.5 and persist to a cosine Chroma."""
    for d in docs:  # Clean metadata (Chroma can't store None values)
        if d.metadata.get("book_page") is None:  # Remove missing book_page keys
            d.metadata.pop("book_page", None)

    if config.CHROMA_DIR.exists():  # Remove old store if rebuilding
        print(f"      removing existing store at {config.CHROMA_DIR} (rebuild)")
        shutil.rmtree(config.CHROMA_DIR)

    print(f"[4/4] Embedding {len(docs)} chunks with {config.EMBEDDING_MODEL} ...")
    print("      (first run downloads the model, ~1.3 GB; CPU embedding is slow)")
    store = Chroma(  # Create Chroma vector store
        collection_name=COLLECTION,  # Collection name
        embedding_function=get_embeddings(),  # BGE embeddings
        persist_directory=str(config.CHROMA_DIR),  # Save location
        collection_metadata={"hnsw:space": "cosine"},  # Use cosine similarity (paper's choice)
    )

    for i in range(0, len(docs), EMBED_BATCH):  # Embed in batches
        batch = docs[i : i + EMBED_BATCH]  # Get batch
        store.add_documents(batch)  # Embed and add to Chroma
        print(f"      embedded {min(i + EMBED_BATCH, len(docs))}/{len(docs)}")

    print(f"      persisted -> {config.CHROMA_DIR}  (collection '{COLLECTION}')")
    return store


def main() -> None:  # Run full Phase A pipeline
    docs = run_ingestion()  # Extract, clean, chunk
    if docs:  # Show a sample chunk
        sample = docs[len(docs) // 2]  # Get middle chunk
        print("\nSample chunk metadata:", sample.metadata)

    store = build_vector_store(docs)  # Embed and persist to Chroma

    q = "What is a functional unit in LCA?"  # Test query
    hits = store.similarity_search(q, k=3)  # Retrieve top-3 similar chunks
    print(f"\nSanity query: {q!r}")
    for h in hits:  # Display results
        meta = h.metadata
        print(f"  -> {meta.get('chapter')} | pdf p.{meta.get('pdf_page')}")
        print(f"     {h.page_content[:120]!r}")


if __name__ == "__main__":
    main()
