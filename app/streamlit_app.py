"""Streamlit chat UI (Step 6).

Features to replicate from the paper's Fig. 8:
  1. Chat box — ask a question, get a grounded answer
  2. Conversation history — re-sent so follow-ups stay in context
  3. PDF upload — chunk/embed/add to the knowledge base live
  4. Source display — show the chapter/page each answer came from

Usage (once implemented):  streamlit run app/streamlit_app.py

TODO: implement the chat interface (calls phase_b_query.rag_pipeline.answer)
"""
