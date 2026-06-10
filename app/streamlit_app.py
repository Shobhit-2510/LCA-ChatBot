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

# Fix sys.path so imports work (Streamlit adds app/ but not project root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # added root directory to path for imports

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from phase_b_query.rag_pipeline import answer

# Configure page metadata
st.set_page_config(page_title="Chat-LCA", page_icon="🤖")
st.title("Chat-LCA")  # Page title
st.caption("powered by Hauschild et al. — *LCA: Theory and Practice*")  # Subtitle


def render_sources(sources: list[dict]) -> None:
    """Display source citations (chapter/page) in a collapsible section."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):  # Collapsible sources box
        for s in sources:
            page = s.get("book_page") or s.get("pdf_page")  # Use book page if available
            st.markdown(f"- **{s.get('chapter', 'Unknown')}** — p.{page}")  # Show as bullet


def to_lc_history(messages: list[dict]) -> list:
    """Convert chat history to LangChain message objects for context."""
    history = []
    for m in messages:  # Iterate through conversation history
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))  # User turn
        else:
            history.append(AIMessage(content=m["content"]))  # Assistant turn
    return history


# Initialize conversation history in session state (persists across reruns)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior conversation turns
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):  # User or assistant message bubble
        st.markdown(msg["content"])  # Show message text
        if msg["role"] == "assistant":  # If assistant, also show sources
            render_sources(msg.get("sources", []))

# Handle new question from user
question = st.chat_input("Ask a question about LCA…")  # Input box for user question
if question:
    # Show user's question immediately
    with st.chat_message("user"): # create a user message bubble
        st.markdown(question) # display the user's question in the bubble

    # Convert prior turns to LangChain format for context (history excludes this question)
    history = to_lc_history(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question}) # Save user question to history (for next turn's context)

    # Get RAG answer with loading indicator
    with st.chat_message("assistant"):
        with st.spinner("Searching the textbook…"):
            result = answer(question, history=history)  # Call RAG pipeline
        st.markdown(result["answer"])  # Display answer
        render_sources(result["sources"])  # Display sources

    # Save assistant response to history (for next turn's context)
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
