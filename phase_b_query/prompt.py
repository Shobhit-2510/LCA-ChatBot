"""Prompt template (Step 5).

The grounding prompt the LLM receives: system role + retrieved context +
user question, instructing answer-only-from-context with chapter/page cited.

Template (from the plan):
    You are an LCA expert assistant.
    Answer ONLY from the context below.
    If the answer is not there, say so.

    Context:
    {context}

    Question:
    {question}

    Answer with the chapter/page cited.

TODO: implement build_prompt() -> PromptTemplate
"""
