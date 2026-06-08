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

PDF_PATH = config.DATA_RAW / "Textbook.pdf"
CHUNKS_OUT = config.DATA_PROCESSED / "chunks.jsonl"
COLLECTION = "lca_book"
EMBED_BATCH = 256


def run_ingestion() -> list:
    if not PDF_PATH.exists():
        sys.exit(f"PDF not found: {PDF_PATH}")

    print(f"[1/3] Extracting text from {PDF_PATH.name} ...")
    pages, chapter_index = extract(str(PDF_PATH))
    print(f"      {len(pages)} pages, {len(chapter_index)} top-level TOC entries")

    print("[2/3] Cleaning headers/footers, hyphenation, ligatures ...")
    cleaned = clean_pages(pages, chapter_index)
    print(f"      {len(cleaned)} non-empty pages after cleaning")

    print(f"[3/3] Chunking (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}) ...")
    docs = chunk_pages(cleaned, chapter_index)
    print(f"      {len(docs)} chunks produced")

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_OUT, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"text": d.page_content, "metadata": d.metadata}) + "\n")
    print(f"      saved -> {CHUNKS_OUT}")

    return docs


def build_vector_store(docs: list) -> Chroma:
    """Embed chunks with bge-large-en-v1.5 and persist to a cosine Chroma."""
    # Chroma can't store None metadata values; drop missing book_page keys.
    for d in docs:
        if d.metadata.get("book_page") is None:
            d.metadata.pop("book_page", None)

    if config.CHROMA_DIR.exists():
        print(f"      removing existing store at {config.CHROMA_DIR} (rebuild)")
        shutil.rmtree(config.CHROMA_DIR)

    print(f"[4/4] Embedding {len(docs)} chunks with {config.EMBEDDING_MODEL} ...")
    print("      (first run downloads the model, ~1.3 GB; CPU embedding is slow)")
    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},  # paper's similarity
    )

    for i in range(0, len(docs), EMBED_BATCH):
        batch = docs[i : i + EMBED_BATCH]
        store.add_documents(batch)
        print(f"      embedded {min(i + EMBED_BATCH, len(docs))}/{len(docs)}")

    print(f"      persisted -> {config.CHROMA_DIR}  (collection '{COLLECTION}')")
    return store


def main() -> None:
    docs = run_ingestion()
    if docs:
        sample = docs[len(docs) // 2]
        print("\nSample chunk metadata:", sample.metadata)

    store = build_vector_store(docs)

    # Sanity retrieval to confirm the store answers queries.
    q = "What is a functional unit in LCA?"
    hits = store.similarity_search(q, k=3)
    print(f"\nSanity query: {q!r}")
    for h in hits:
        meta = h.metadata
        print(f"  -> {meta.get('chapter')} | pdf p.{meta.get('pdf_page')}")
        print(f"     {h.page_content[:120]!r}")


if __name__ == "__main__":
    main()
