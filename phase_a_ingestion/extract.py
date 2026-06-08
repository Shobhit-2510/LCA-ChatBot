"""Step 1 — extract raw text from the book PDF.

PyMuPDF (fitz) reads page by page; pdfplumber is the fallback for tables.
Each page's text is kept with its page number so citations survive.

TODO: implement extract_pdf(pdf_path) -> list[{"page": int, "text": str}]
"""
