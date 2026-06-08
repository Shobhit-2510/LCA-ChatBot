"""Phase A driver — extract -> clean -> chunk (-> embed -> Chroma: next step).

Currently runs ingestion through chunking and writes the tagged chunks to
data/processed/chunks.jsonl for inspection. Embedding + Chroma persistence
is the next stage (see TODO below).

Usage:  python -m phase_a_ingestion.build_index
"""

from __future__ import annotations

import json
import sys

import config
from phase_a_ingestion.extract import extract
from phase_a_ingestion.clean import clean_pages
from phase_a_ingestion.chunk import chunk_pages

PDF_PATH = config.DATA_RAW / "Textbook.pdf"
CHUNKS_OUT = config.DATA_PROCESSED / "chunks.jsonl"


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


def main() -> None:
    docs = run_ingestion()
    # Quick sanity peek
    if docs:
        sample = docs[len(docs) // 2]
        print("\nSample chunk metadata:", sample.metadata)
        print("Sample text (first 200 chars):", repr(sample.page_content[:200]))

    # TODO (next step): embed with phase_a_ingestion.embed.get_embeddings()
    #                   and persist to Chroma at config.CHROMA_DIR.


if __name__ == "__main__":
    main()
