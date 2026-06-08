"""Step 3 — split cleaned text into overlapping chunks.

LangChain RecursiveCharacterTextSplitter with the paper's settings
(CHUNK_SIZE=1000, CHUNK_OVERLAP=200 from config). Pages are grouped per
chapter and concatenated so chunks respect chapter boundaries but can span
page breaks; each chunk is mapped back to the page where it starts, so
chapter + page metadata travels with it for citations.

Public API:
    chunk_pages(cleaned_pages, chapter_index) -> list[Document]
"""

from __future__ import annotations

import bisect
import re
from itertools import groupby

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from phase_a_ingestion.extract import chapter_for_page

_SEP = "\n\n"  # page joiner inside a chapter


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _assemble_chapter(pages: list[dict]) -> tuple[str, list[int], list[dict]]:
    """Join a chapter's pages; return (text, page_start_offsets, pages)."""
    parts, offsets, cursor = [], [], 0
    for page in pages:
        offsets.append(cursor)
        parts.append(page["text"])
        cursor += len(page["text"]) + len(_SEP)
    return _SEP.join(parts), offsets, pages


def chunk_pages(
    cleaned_pages: list[dict], chapter_index: list[tuple[int, str]]
) -> list[Document]:
    """Split cleaned pages into Documents tagged with chapter + page."""
    splitter = _splitter()
    docs: list[Document] = []
    chunk_id = 0

    # Group consecutive pages by their chapter (pages are already in order).
    keyed = [
        (chapter_for_page(chapter_index, p["pdf_page"]), p) for p in cleaned_pages
    ]
    for chapter, group in groupby(keyed, key=lambda kp: kp[0]):
        pages = [p for _, p in group]
        text, page_offsets, pages = _assemble_chapter(pages)

        cursor = 0
        for raw in splitter.split_text(text):
            # Locate this chunk's start offset (on the raw text) to find its
            # source page, before trimming the leading separator artifact.
            start = text.find(raw, max(0, cursor - config.CHUNK_OVERLAP))
            if start == -1:
                start = cursor
            cursor = start + len(raw)

            # The ". " separator stays attached to the front of the next
            # chunk; trim the dangling leading punctuation/whitespace.
            content = re.sub(r"^[\s.]+", "", raw).strip()
            if not content:
                continue

            pi = bisect.bisect_right(page_offsets, start) - 1
            pi = max(0, pi)
            src = pages[pi]

            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "chunk_id": chunk_id,
                        "chapter": chapter,
                        "pdf_page": src["pdf_page"],
                        "book_page": src["book_page"],
                        "source": "Hauschild et al., LCA: Theory and Practice",
                    },
                )
            )
            chunk_id += 1
    return docs
