"""RAG pipeline — wires retriever + prompt + LLM into one answer() call.

The Phase B entry point used by both the Streamlit UI and the evaluation
harness. Returns the grounded answer plus its source chunks (chapter/page)
for citation display.

TODO: implement answer(question, history=None) -> {"answer": str, "sources": list}
"""
