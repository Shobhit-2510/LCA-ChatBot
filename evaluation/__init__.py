"""Evaluation — build a QA test set and grade the chatbot.

Generate Q&A pairs from the book, label & verify (Cohen's Kappa), then
score answers with BERTScore / cosine / ROUGE-L vs. a no-RAG baseline.
"""
