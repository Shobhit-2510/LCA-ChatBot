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

# Placeholder for empty/blank model answers. An empty string is a legitimate
# (bad) answer — e.g. the model returned nothing — but bert-score's empty-text
# path crashes on transformers 5.x, so map it to a sentinel that scores low
# rather than erroring.
_EMPTY = "[no answer]"


def _safe(texts: list[str]) -> list[str]:
    return [t if t and t.strip() else _EMPTY for t in texts]


def bertscore_f1(predictions: list[str], references: list[str]) -> float:
    """Mean BERTScore F1 (semantic similarity).

    Raw F1 (no baseline rescaling) — this is the scale the paper reports
    (~0.85) and the >=0.80 target is measured on, where good answers sit
    around 0.85-0.92. Rescaling would put scores on a different, lower scale.
    """
    from bert_score import score

    _, _, f1 = score(_safe(predictions), _safe(references), lang="en")
    return float(f1.mean())


def cosine_sim(predictions: list[str], references: list[str]) -> float:
    """Mean cosine similarity of answer embeddings (bge-large-en-v1.5)."""
    from phase_a_ingestion.embed import get_embeddings

    emb = get_embeddings()
    p = np.array(emb.embed_documents(_safe(predictions)))
    r = np.array(emb.embed_documents(_safe(references)))
    # Embeddings are L2-normalized, so the row-wise dot product is the cosine.
    sims = np.sum(p * r, axis=1) / (
        np.linalg.norm(p, axis=1) * np.linalg.norm(r, axis=1)
    )
    return float(sims.mean())


def rouge_l(predictions: list[str], references: list[str]) -> float:
    """Mean ROUGE-L F-measure (longest common subsequence overlap)."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(ref, pred)["rougeL"].fmeasure
        for pred, ref in zip(predictions, references)
    ]
    return float(np.mean(scores)) if scores else 0.0


def evaluate(predictions: list[str], references: list[str]) -> dict:
    """Run all three metrics; return a dict of mean scores."""
    return {
        "bertscore_f1": bertscore_f1(predictions, references),
        "cosine": cosine_sim(predictions, references),
        "rouge_l": rouge_l(predictions, references),
    }
