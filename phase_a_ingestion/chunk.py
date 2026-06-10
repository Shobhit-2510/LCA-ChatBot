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

_SEP = "\n\n"  # Separator joining pages within a chapter to make a single string  


def _splitter() -> RecursiveCharacterTextSplitter:
    """Create text splitter with paper's exact settings."""
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,  # Chunk size in characters
        chunk_overlap=config.CHUNK_OVERLAP,  # Overlap between chunks
        separators=["\n\n", "\n", ". ", " ", ""],  # Split prioritizing paragraphs first, A paragraph is better than a sentence is better than a word.
    )


def _assemble_chapter(pages: list[dict]) -> tuple[str, list[int], list[dict]]:
    """Join a chapter's pages; return (text, page_start_offsets, pages)."""
    parts, offsets, cursor = [], [], 0 # offsets = start position of each page in the assembled text
    for page in pages:
        offsets.append(cursor)  # Record start position of this page in assembled text
        parts.append(page["text"])  # Add page text
        cursor += len(page["text"]) + len(_SEP)  # Update cursor for next page
    return _SEP.join(parts), offsets, pages  # Return joined text (string), offsets, and original page list


def chunk_pages(
    cleaned_pages: list[dict], chapter_index: list[tuple[int, str]]
) -> list[Document]:
    """Split cleaned pages into Documents tagged with chapter + page."""
    splitter = _splitter()
    docs: list[Document] = []
    chunk_id = 0

    # keyed is list of (chapter_title, page_dict) tuples
    keyed = [
        (chapter_for_page(chapter_index, p["pdf_page"]), p) for p in cleaned_pages
    ]
    # groupby walks through keyed and groups consecutive elements with the same key: returns (chapter_title, group) where group is tuples of (chapter_title, page_dict) for that chapter.
    for chapter, group in groupby(keyed, key=lambda kp: kp[0]):
        pages = [p for _, p in group]  # Extract pages for this chapter
        text, page_offsets, pages = _assemble_chapter(pages)  # Join pages with separator

        cursor = 0 # start position of the current chunk
        # raw = (chunk text without overlap) for each chunk produced by the splitter.
        for raw in splitter.split_text(text):  # Split text into chunks
            # Find chunk's start position in assembled text to identify source page
            start = text.find(raw, max(0, cursor - config.CHUNK_OVERLAP))
            if start == -1:
                start = cursor
            cursor = start + len(raw) # start position of the next chunk

            # Remove leading separators/punctuation artifact from splitting
            content = re.sub(r"^[\s.]+", "", raw).strip()
            if not content:  # Skip empty chunks
                continue

            # Find which page this chunk came from using binary search on page offsets (offsets = start position of each page in the assembled text)
            pi = bisect.bisect_right(page_offsets, start) - 1
            pi = max(0, pi) # Clamp to valid page index in case of edge cases
            src = pages[pi] # Source page dict for this chunk (contains pdf_page, book_page, text)

            # Create Document with chunk content and metadata (chapter, page, etc.)
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
