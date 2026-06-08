"""Step 1 — extract raw text from the book PDF.

PyMuPDF (fitz) reads page by page. The book ships with a rich embedded
table of contents, so chapters are mapped from the TOC (reliable) rather
than parsed from running headers (fragile).

Public API:
    extract(pdf_path) -> (pages, chapter_index)
        pages         : list[{"pdf_page": int, "text": str}]   # pdf_page is 1-based
        chapter_index : list[(start_pdf_page, chapter_title)]   # sorted ascending
    chapter_for_page(chapter_index, pdf_page) -> str
"""

from __future__ import annotations

import bisect

import fitz  # PyMuPDF


def build_chapter_index(doc: fitz.Document) -> list[tuple[int, str]]:
    """Map PDF pages to chapter titles using the embedded TOC.

    Uses top-level (level 1) TOC entries. Anything before the first numbered
    chapter (Preface, Contents, Editors...) is labelled "Front Matter".
    TOC page numbers from PyMuPDF are 1-based PDF page numbers.
    """
    index: list[tuple[int, str]] = []
    for level, title, page in doc.get_toc():
        if level == 1:
            index.append((page, title.strip()))
    index.sort(key=lambda x: x[0])
    return index


def chapter_for_page(chapter_index: list[tuple[int, str]], pdf_page: int) -> str:
    """Return the chapter title whose span contains `pdf_page` (1-based)."""
    if not chapter_index:
        return "Unknown"
    starts = [start for start, _ in chapter_index]
    pos = bisect.bisect_right(starts, pdf_page) - 1
    if pos < 0:
        return "Front Matter"
    return chapter_index[pos][1]


def extract(pdf_path: str) -> tuple[list[dict], list[tuple[int, str]]]:
    """Read the PDF page by page; return raw page texts + chapter index."""
    doc = fitz.open(pdf_path)
    chapter_index = build_chapter_index(doc)
    pages: list[dict] = []
    for i in range(doc.page_count):
        text = doc[i].get_text("text")  # plain reading-order text
        pages.append({"pdf_page": i + 1, "text": text})
    doc.close()
    return pages, chapter_index
