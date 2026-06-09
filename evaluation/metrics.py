"""Scoring metrics (Evaluation Part 2).

Three metrics from the plan, comparing a model answer to its reference:
  * BERTScore  — semantic match via a neural model (Eq. 4/6); target >= 0.80
  * Cosine     — embed both answers, measure the angle (Eq. 3); sanity check
  * ROUGE-L    — word overlap via longest common subsequence (Eq. 7); strict

Each function takes parallel lists (predictions, references) and returns the
mean score; evaluate() runs all three at once.

Note: the first BERTScore call downloads roberta-large (~1.4 GB).
"""

from __future__ import annotations

import numpy as np

# Sentinel for empty answers (works around bert-score bug in transformers 5.x)
_EMPTY = "[no answer]"


def _safe(texts: list[str]) -> list[str]:
    """Replace empty strings with placeholder to avoid bert-score crash."""
    return [t if t and t.strip() else _EMPTY for t in texts]


def bertscore_f1(predictions: list[str], references: list[str]) -> float:
    """Mean BERTScore F1 (semantic similarity from BERT embeddings)."""
    from bert_score import score

    _, _, f1 = score(_safe(predictions), _safe(references), lang="en")  # Get F1 scores
    return float(f1.mean())  # Return mean across all pairs


def cosine_sim(predictions: list[str], references: list[str]) -> float:
    """Mean cosine similarity using BGE embeddings."""
    from phase_a_ingestion.embed import get_embeddings

    emb = get_embeddings()  # Get BGE embedding model
    p = np.array(emb.embed_documents(_safe(predictions)))  # Embed predictions
    r = np.array(emb.embed_documents(_safe(references)))  # Embed references
    # Compute cosine similarity (dot product of L2-normalized vectors)
    sims = np.sum(p * r, axis=1) / (
        np.linalg.norm(p, axis=1) * np.linalg.norm(r, axis=1)
    )
    return float(sims.mean())  # Return mean cosine similarity


def rouge_l(predictions: list[str], references: list[str]) -> float:
    """Mean ROUGE-L (word overlap via longest common subsequence)."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)  # Create ROUGE scorer
    scores = [
        scorer.score(ref, pred)["rougeL"].fmeasure  # Compute F-measure per pair
        for pred, ref in zip(predictions, references)
    ]
    return float(np.mean(scores)) if scores else 0.0  # Return mean


def evaluate(predictions: list[str], references: list[str]) -> dict:
    """Compute all three metrics: BERTScore, cosine, and ROUGE-L."""
    return {
        "bertscore_f1": bertscore_f1(predictions, references),  # Semantic similarity
        "cosine": cosine_sim(predictions, references),  # Embedding cosine distance
        "rouge_l": rouge_l(predictions, references),  # Word overlap
    }
