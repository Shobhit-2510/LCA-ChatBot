"""Prompt template (Step 5) — the grounding prompt from slide 9.

The prompt from slide 9 is split into two parts:
- System message: tells the LLM what role to play and how to behave
- Human message: contains the retrieved context and user's question
This two-part format is the standard way to structure chat prompts.

The instruction "respond with only the final answer" is important: without it,
Claude might show its thinking process in the answer, which we don't want.

Public API:
    PROMPT : A chat prompt template with {context} and {question} placeholders
             that you fill in with the actual context and user question.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

# System message: defines role and rules for the LLM
SYSTEM = (
    "You are an LCA expert assistant.\n"
    "Answer ONLY from the context below.\n"
    "If the answer is not there, say so.\n"
    "Respond with only the final answer — do not narrate your reasoning."
) # these all are combined into a single string with newlines.

# Human message: context (retrieved chunks) + question, with citation instruction
HUMAN = (
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Answer with the chapter/page cited."
)

# Combine system + human messages into a template with {context} and {question} placeholders
PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM), ("human", HUMAN)]
)
