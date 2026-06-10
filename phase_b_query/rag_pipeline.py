"""RAG pipeline — wires retriever + prompt + LLM into one answer() call.

The Phase B entry point used by both the Streamlit UI and the evaluation
harness. Retrieves the top-10 MMR chunks, grounds Claude on them, and
returns the answer plus its source chunks (chapter/page) for citation.

Public API:
    answer(question, history=None) -> {"answer": str, "sources": list[dict]}
    format_context(docs) -> str
"""

from __future__ import annotations

from langchain_core.documents import Document

from phase_b_query.retriever import get_retriever
from phase_b_query.prompt import PROMPT
from phase_b_query.llm import get_llm


def _cite(meta: dict) -> str: # meta = {'chapter': 'Chapter 1: Introduction to LCA', 'pdf_page': 5, 'book_page': 3}
    """Human-readable page label, preferring the printed book page."""
    page = meta.get("book_page") or meta.get("pdf_page")  # Use book page if available
    return f"{meta.get('chapter', 'Unknown')}, p.{page}"  # Format as "Chapter_name, p.X"


def format_context(docs: list[Document]) -> str:
    """Lay out retrieved chunks with source tags so Claude can cite them."""
    # Convert list of chunks into a single formatted string with source labels
    return "\n\n".join(  # Join all chunks separated by double newlines for readability
        f"[Source {i}: {_cite(d.metadata)}]\n{d.page_content}"  # Format: [Source 1: Chapter, p.X]\nchunk text
        for i, d in enumerate(docs, start=1)  # Loop through each document, numbering from 1
    )


def _sources(docs: list[Document]) -> list[dict]:
    """Unique source references, order-preserved, for display.

    Input:

    docs = [
    Document(..., metadata={"chapter": "Basics", "pdf_page": 10, "book_page": 5}),
    Document(..., metadata={"chapter": "Basics", "pdf_page": 10, "book_page": 5}),  # Duplicate!
    Document(..., metadata={"chapter": "Methods", "pdf_page": 25, "book_page": 20}),
    ]

    Output:
    [
        {"chapter": "Basics", "pdf_page": 10, "book_page": 5},
        {"chapter": "Methods", "pdf_page": 25, "book_page": 20},
    ]
"""
    seen, out = set(), []
    for d in docs:
        m = d.metadata
        key = (m.get("chapter"), m.get("pdf_page"))  # Use (chapter, page) as unique key
        if key not in seen:  # Avoid duplicates
            seen.add(key)
            out.append({  # Add source metadata
                "chapter": m.get("chapter"),
                "pdf_page": m.get("pdf_page"),
                "book_page": m.get("book_page"),
            })
    return out


def _text(content) -> str:
    """Extract answer text from an AIMessage (string or block list)."""
    if isinstance(content, str):  # Simple case: content is plain string
        return content.strip() # return copy of string with leading and trailing whitespace removed.
    # Complex case: content is list of blocks with type and text fields
    parts = [
        b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text" # Extract text from blocks of type "text"
    ]
    return "".join(parts).strip()  # Join all text parts in single string and trim whitespace


def answer(question: str, history: list | None = None) -> dict:
    """Answer a question grounded in the retrieved book chunks."""
    docs = get_retriever().invoke(question)  # Retrieve top-10 MMR chunks
    context = format_context(docs)  # Format chunks with source tags

    messages = PROMPT.format_messages(context=context, question=question)  # Fill prompt template
    if history:  # Include conversation history if provided
        messages = [messages[0], *history, messages[1]]  # Inject history between system and human # messages[0] is system message, messages[1] is human message, history goes in between

    response = get_llm().invoke(messages)  # Call Claude to generate answer
    return {"answer": _text(response.content), "sources": _sources(docs)}  # Return answer + sources
