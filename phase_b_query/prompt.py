"""Prompt template (Step 5) — the grounding prompt from slide 9.

Slide 9's single prompt is split into a system message (role + grounding
rules) and a human message (retrieved context + question), which is the
idiomatic shape for a chat model.

The final-answer-only line is deliberate: with extended thinking off,
Opus 4.8 can otherwise spill reasoning into the visible answer.

Public API:
    PROMPT : ChatPromptTemplate  with {context} and {question} variables
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

SYSTEM = (
    "You are an LCA expert assistant.\n"
    "Answer ONLY from the context below.\n"
    "If the answer is not there, say so.\n"
    "Respond with only the final answer — do not narrate your reasoning."
)

HUMAN = (
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Answer with the chapter/page cited."
)

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM), ("human", HUMAN)]
)
