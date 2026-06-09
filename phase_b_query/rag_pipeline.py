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


def _cite(meta: dict) -> str:
    """Human-readable page label, preferring the printed book page."""
    page = meta.get("book_page") or meta.get("pdf_page")  # Use book page if available
    return f"{meta.get('chapter', 'Unknown')}, p.{page}"  # Format as "Chapter, p.X"


def format_context(docs: list[Document]) -> str:
    """Lay out retrieved chunks with source tags so Claude can cite them."""
    return "\n\n".join(
        f"[Source {i}: {_cite(d.metadata)}]\n{d.page_content}"  # Tag each chunk with source
        for i, d in enumerate(docs, start=1)
    )


def _sources(docs: list[Document]) -> list[dict]:
    """Unique source references, order-preserved, for display."""
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
        return content.strip()
    # Complex case: content is list of blocks with type and text fields
    parts = [
        b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
    ]
    return "".join(parts).strip()  # Join all text parts


def answer(question: str, history: list | None = None) -> dict:
    """Answer a question grounded in the retrieved book chunks."""
    docs = get_retriever().invoke(question)  # Retrieve top-10 MMR chunks
    context = format_context(docs)  # Format chunks with source tags

    messages = PROMPT.format_messages(context=context, question=question)  # Fill prompt template
    if history:  # Include conversation history if provided
        messages = [messages[0], *history, messages[1]]  # Inject history between system and human

    response = get_llm().invoke(messages)  # Call Claude to generate answer
    return {"answer": _text(response.content), "sources": _sources(docs)}  # Return answer + sources
