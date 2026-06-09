"""Step 2 — clean extracted text.

Removes the book's repeated headers/footers (page numbers, the
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

_LIGATURES = {  # Map fancy typography glyphs to ASCII equivalents for consistent text processing
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl",
    "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "ft", "ﬆ": "st",
}

_AUTHOR_FOOTER = re.compile(r"^[A-Z]\.\s+[\w’’\-]+(?:\s+et\s+al\.?)$", re.UNICODE)  # Matches author footer like "A. Bjørn et al."
_BARE_INT = re.compile(r"^\d{1,4}$")  # Matches a bare page number (1-4 digits alone on a line)
_NUMBERED_HEADER = re.compile(r"^\d+\s+\S")  # Matches running headers like "2  Main Characteristics of LCA"


def _normalize_glyphs(text: str) -> str:
    """Convert all fancy typography glyphs to standard ASCII equivalents."""
    for lig, repl in _LIGATURES.items():  # Replace each ligature with ASCII equivalent
        text = text.replace(lig, repl)
    text = unicodedata.normalize("NFKC", text)  # Normalize remaining fancy characters to ASCII
    return text


def _is_running_header(line: str, chapter_titles: set[str]) -> bool:
    """Check if a line is a repeating chapter header (noise to remove from pages)."""
    s = line.strip()  # Remove leading/trailing whitespace
    if s in chapter_titles:  # Check if line matches a known chapter title exactly
        return True
    if _NUMBERED_HEADER.match(s):  # Check if line is "number title" pattern
        stripped = re.sub(r"^\d+\s+", "", s)  # Remove leading number
        if stripped in {re.sub(r"^\d+\s+", "", t) for t in chapter_titles}:  # Match stripped title
            return True
    return False


def clean_page_text(
    text: str, chapter_titles: set[str]
) -> tuple[str, int | None]:
    """Clean one page; return (clean_text, recovered_book_page)."""
    text = _normalize_glyphs(text)  # Convert fancy ligatures to ASCII
    lines = text.split("\n")  # Split page into individual lines for filtering

    book_page: int | None = None  # Store the printed book page number from footer
    kept: list[str] = []  # Accumulate lines that pass all filters

    for line in lines:  # Iterate through lines and remove noise
        s = line.strip()  # Remove leading/trailing whitespace
        if not s:  # Skip empty lines
            continue
        if _BARE_INT.match(s):  # Detect and extract lone page numbers like "23"
            book_page = int(s)  # A lone number is the printed page number
            continue  # Don't include page number in body text
        if _AUTHOR_FOOTER.match(s):  # Skip author footer lines
            continue
        if _is_running_header(s, chapter_titles):  # Skip repeated chapter headers
            continue
        kept.append(s)  # Keep all other lines

    body = "\n".join(kept)  # Rejoin kept lines with newlines
    body = re.sub(r"(\w)-\n(\w)", r"\1\2", body)  # Fix hyphenation broken across lines
    body = re.sub(r"\n{2,}", "\n\n", body)  # Collapse 2+ newlines to exactly 2 (preserve paragraphs)
    body = re.sub(r"(?<!\n)\n(?!\n)", " ", body)  # Convert single newlines to spaces
    body = re.sub(r"[ \t]{2,}", " ", body).strip()  # Collapse multiple spaces/tabs
    return body, book_page


def clean_pages(
    pages: list[dict], chapter_index: list[tuple[int, str]]
) -> list[dict]:
    """Clean every page, attaching the recovered printed page number."""
    chapter_titles = {title for _, title in chapter_index}  # Extract chapter titles
    cleaned: list[dict] = []  # Accumulate cleaned pages
    for page in pages:  # Process each page
        body, book_page = clean_page_text(page["text"], chapter_titles)  # Clean one page
        if not body:  # Skip if page became empty after cleaning
            continue
        cleaned.append({  # Add cleaned page with metadata
            "pdf_page": page["pdf_page"],
            "book_page": book_page,
            "text": body,
        })
    return cleaned
