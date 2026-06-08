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
    page = meta.get("book_page") or meta.get("pdf_page")
    return f"{meta.get('chapter', 'Unknown')}, p.{page}"


def format_context(docs: list[Document]) -> str:
    """Lay out retrieved chunks with source tags so Claude can cite them."""
    return "\n\n".join(
        f"[Source {i}: {_cite(d.metadata)}]\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )


def _sources(docs: list[Document]) -> list[dict]:
    """Unique source references, order-preserved, for display."""
    seen, out = set(), []
    for d in docs:
        m = d.metadata
        key = (m.get("chapter"), m.get("pdf_page"))
        if key not in seen:
            seen.add(key)
            out.append(
                {
                    "chapter": m.get("chapter"),
                    "pdf_page": m.get("pdf_page"),
                    "book_page": m.get("book_page"),
                }
            )
    return out


def _text(content) -> str:
    """Extract answer text from an AIMessage (string or block list)."""
    if isinstance(content, str):
        return content.strip()
    parts = [
        b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
    ]
    return "".join(parts).strip()


def answer(question: str, history: list | None = None) -> dict:
    """Answer a question grounded in the retrieved book chunks."""
    docs = get_retriever().invoke(question)
    context = format_context(docs)

    messages = PROMPT.format_messages(context=context, question=question)
    if history:  # inject prior turns between system and current question
        messages = [messages[0], *history, messages[1]]

    response = get_llm().invoke(messages)
    return {"answer": _text(response.content), "sources": _sources(docs)}
