"""Step 2 — clean extracted text.

Removes the book's running headers/footers (page numbers, the
"A. Bjørn et al." author footer, running chapter titles), repairs
hyphenation broken across line breaks, and normalizes ligatures.

While stripping, the printed page number is recovered from the footer so
answers can cite the *book's* page, not just the PDF index.

Public API:
    clean_pages(pages, chapter_index) -> list[{"pdf_page", "book_page", "text"}]
"""

from __future__ import annotations

import re
import unicodedata

# Ligatures PyMuPDF leaves as single glyphs -> ASCII equivalents.
_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl",
    "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "ft", "ﬆ": "st",
}

# Author running footer, e.g. "A. Bjørn et al." — any "<Name> et al." line.
_AUTHOR_FOOTER = re.compile(r"^[A-Z]\.\s+[\w’'\-]+(?:\s+et\s+al\.?)$", re.UNICODE)
_BARE_INT = re.compile(r"^\d{1,4}$")
# A running header like "2  Main Characteristics of LCA" (chapter no. + title).
_NUMBERED_HEADER = re.compile(r"^\d+\s+\S")


def _normalize_glyphs(text: str) -> str:
    for lig, repl in _LIGATURES.items():
        text = text.replace(lig, repl)
    # Normalize remaining compatibility glyphs and odd whitespace.
    text = unicodedata.normalize("NFKC", text)
    return text


def _is_running_header(line: str, chapter_titles: set[str]) -> bool:
    """True if the line is a running chapter header (page-top/bottom noise)."""
    s = line.strip()
    if s in chapter_titles:
        return True
    # "<n> <Chapter Title>" — match against known titles ignoring the number.
    if _NUMBERED_HEADER.match(s):
        stripped = re.sub(r"^\d+\s+", "", s)
        if stripped in {re.sub(r"^\d+\s+", "", t) for t in chapter_titles}:
            return True
    return False


def clean_page_text(
    text: str, chapter_titles: set[str]
) -> tuple[str, int | None]:
    """Clean one page; return (clean_text, recovered_book_page)."""
    text = _normalize_glyphs(text)
    lines = text.split("\n")

    book_page: int | None = None
    kept: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if _BARE_INT.match(s):
            # A lone number near header/footer is the printed page number.
            book_page = int(s)
            continue
        if _AUTHOR_FOOTER.match(s):
            continue
        if _is_running_header(s, chapter_titles):
            continue
        kept.append(s)

    body = "\n".join(kept)
    # Repair hyphenation broken across a line break: "char-\nacteristics".
    body = re.sub(r"(\w)-\n(\w)", r"\1\2", body)
    # Remaining single newlines -> spaces (keep paragraph blank lines).
    body = re.sub(r"\n{2,}", "\n\n", body)
    body = re.sub(r"(?<!\n)\n(?!\n)", " ", body)
    body = re.sub(r"[ \t]{2,}", " ", body).strip()
    return body, book_page


def clean_pages(
    pages: list[dict], chapter_index: list[tuple[int, str]]
) -> list[dict]:
    """Clean every page, attaching the recovered printed page number."""
    chapter_titles = {title for _, title in chapter_index}
    cleaned: list[dict] = []
    for page in pages:
        body, book_page = clean_page_text(page["text"], chapter_titles)
        if not body:
            continue  # blank / fully-stripped page
        cleaned.append(
            {
                "pdf_page": page["pdf_page"],
                "book_page": book_page,
                "text": body,
            }
        )
    return cleaned
