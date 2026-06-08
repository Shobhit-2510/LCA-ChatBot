"""Streamlit chat UI (Step 6).

A grounded chat over the Hauschild LCA textbook. Features implemented:
  1. Chat box — ask a question, get a grounded answer
  2. Conversation history — prior turns are re-sent so follow-ups stay in context
  4. Source display — the chapter/page each answer came from

(PDF upload, feature 3, is intentionally out of scope here.)

Usage:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import sys

# Streamlit puts this file's dir (app/) on sys.path, not the project root —
# add the root so the phase_b_query package resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from phase_b_query.rag_pipeline import answer

st.set_page_config(page_title="Chat-LCA", page_icon="📖")
st.title("Chat-LCA")
st.caption("powered by Hauschild et al. — *LCA: Theory and Practice*")


def render_sources(sources: list[dict]) -> None:
    """Show the chapter/page each answer was grounded in."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            page = s.get("book_page") or s.get("pdf_page")
            st.markdown(f"- **{s.get('chapter', 'Unknown')}** — p.{page}")


def to_lc_history(messages: list[dict]) -> list:
    """Convert stored turns to LangChain messages for the LLM context."""
    history = []
    for m in messages:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        else:
            history.append(AIMessage(content=m["content"]))
    return history


if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay the conversation so far.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []))

# New question.
if question := st.chat_input("Ask a question about LCA…"):
    with st.chat_message("user"):
        st.markdown(question)

    # History is every prior turn (excludes the question just asked).
    history = to_lc_history(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Searching the textbook…"):
            result = answer(question, history=history)
        st.markdown(result["answer"])
        render_sources(result["sources"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    )
