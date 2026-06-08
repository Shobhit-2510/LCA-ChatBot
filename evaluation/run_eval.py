"""Evaluation runner — grade RAG vs. a no-RAG baseline.

For each QA pair: get the RAG answer (retrieval + Claude) and a no-RAG
baseline answer (Claude alone, no context), then score both against the
reference with BERTScore / cosine / ROUGE-L. Reproduces the paper's
headline claim: RAG beats the no-RAG baseline, BERTScore >= 0.80.

Usage:
    python -m evaluation.run_eval --limit 20

Output: data/processed/eval_results.json
"""

from __future__ import annotations

import argparse
import json

import config
from phase_b_query.llm import get_llm
from phase_b_query.rag_pipeline import answer, _text
from evaluation import metrics

QA_IN = config.DATA_PROCESSED / "qa_pairs.jsonl"
RESULTS_OUT = config.DATA_PROCESSED / "eval_results.json"

_BASELINE_SYSTEM = (
    "You are an LCA expert assistant. Answer the question concisely from your "
    "own knowledge."
)


def baseline_answer(question: str) -> str:
    """No-RAG answer: Claude alone, no retrieved context."""
    resp = get_llm().invoke(
        [("system", _BASELINE_SYSTEM), ("human", question)]
    )
    return _text(resp.content)


def run(limit: int) -> dict:
    if not QA_IN.exists():
        raise SystemExit(f"Run evaluation.generate_qa first — missing {QA_IN}")
    pairs = [json.loads(l) for l in open(QA_IN, encoding="utf-8")][:limit]
    print(f"Evaluating {len(pairs)} QA pairs (RAG vs no-RAG) ...")

    references, rag_preds, base_preds = [], [], []
    for i, qa in enumerate(pairs, start=1):
        q = qa["question"]
        references.append(qa["reference"])
        rag_preds.append(answer(q)["answer"])
        base_preds.append(baseline_answer(q))
        print(f"  answered {i}/{len(pairs)}")

    print("\nScoring (first BERTScore call downloads roberta-large ~1.4 GB) ...")
    rag_scores = metrics.evaluate(rag_preds, references)
    base_scores = metrics.evaluate(base_preds, references)

    results = {
        "n": len(pairs),
        "rag": rag_scores,
        "no_rag": base_scores,
        "target_bertscore": config.BERTSCORE_TARGET,
    }

    _print_table(results)
    RESULTS_OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved -> {RESULTS_OUT}")
    return results


def _print_table(r: dict) -> None:
    print(f"\n{'metric':<14}{'RAG':>10}{'no-RAG':>10}{'gap':>10}")
    print("-" * 44)
    for k in ("bertscore_f1", "cosine", "rouge_l"):
        gap = r["rag"][k] - r["no_rag"][k]
        print(f"{k:<14}{r['rag'][k]:>10.3f}{r['no_rag'][k]:>10.3f}{gap:>+10.3f}")
    hit = "PASS" if r["rag"]["bertscore_f1"] >= r["target_bertscore"] else "below"
    print(f"\nBERTScore target {r['target_bertscore']}: {hit}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run RAG vs no-RAG evaluation")
    ap.add_argument("--limit", type=int, default=20, help="QA pairs to score")
    args = ap.parse_args()
    run(args.limit)


if __name__ == "__main__":
    main()
