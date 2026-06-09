"""Evaluation runner — grade RAG vs. a no-RAG baseline.

For each QA pair: get the RAG answer (retrieval + Claude) and a no-RAG
baseline answer (Claude alone, no context), then score both against the
reference with BERTScore / cosine / ROUGE-L. Reproduces the paper's
headline claim: RAG beats the no-RAG baseline, BERTScore >= 0.80.

Both answers use the SAME model so the comparison isolates the RAG effect.
That means 2 Opus calls per pair; with a 50 req/min org limit a 240-pair run
takes ~10 min, so answering is rate-limited AND checkpointed — each pair's
answers are appended to eval_predictions.jsonl as they finish, and re-running
resumes from where it left off instead of recomputing.

Usage:
    python -m evaluation.run_eval --limit 240   # safe to re-run to resume

Output:
    data/processed/eval_predictions.jsonl  (per-pair answers; checkpoint)
    data/processed/eval_results.json       (aggregate scores)
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from phase_b_query.llm import get_llm
from phase_b_query.rag_pipeline import answer, _text
from evaluation import metrics

QA_IN = config.DATA_PROCESSED / "qa_pairs.jsonl"
PREDS_OUT = config.DATA_PROCESSED / "eval_predictions.jsonl"
RESULTS_OUT = config.DATA_PROCESSED / "eval_results.json"

MAX_WORKERS = 6
# Org limit is 50 req/min on Opus; cap submissions a touch under that. Both
# the RAG and baseline calls draw from this shared budget.
RATE_PER_MIN = 45

_BASELINE_SYSTEM = (
    "You are an LCA expert assistant. Answer the question concisely from your "
    "own knowledge."
)


class RateLimiter:
    """Thread-safe sliding-window limiter: <= rate calls per 60 s."""

    def __init__(self, rate_per_min: int):
        self.rate = rate_per_min
        self.calls: deque[float] = deque()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                while self.calls and now - self.calls[0] >= 60:
                    self.calls.popleft()
                if len(self.calls) < self.rate:
                    self.calls.append(now)
                    return
                wait = 60 - (now - self.calls[0])
            time.sleep(max(wait, 0.05))


_limiter = RateLimiter(RATE_PER_MIN)


def baseline_answer(question: str) -> str:
    """No-RAG answer: Claude alone, no retrieved context."""
    _limiter.acquire()
    resp = get_llm().invoke([("system", _BASELINE_SYSTEM), ("human", question)])
    return _text(resp.content)


def rag_answer(question: str) -> str:
    _limiter.acquire()
    return answer(question)["answer"]


def _load_checkpoint() -> dict[int, dict]:
    if not PREDS_OUT.exists():
        return {}
    done = {}
    for line in open(PREDS_OUT, encoding="utf-8"):
        rec = json.loads(line)
        done[rec["idx"]] = rec
    return done


def run(limit: int) -> dict:
    if not QA_IN.exists():
        raise SystemExit(f"Run evaluation.generate_qa first — missing {QA_IN}")
    pairs = [json.loads(l) for l in open(QA_IN, encoding="utf-8")][:limit]

    done = _load_checkpoint()
    todo = [(i, qa) for i, qa in enumerate(pairs) if i not in done]
    print(
        f"Evaluating {len(pairs)} QA pairs (RAG vs no-RAG) — "
        f"{len(done)} cached, {len(todo)} to answer, {MAX_WORKERS}-way / {RATE_PER_MIN} rpm"
    )

    if todo:
        # Warm shared singletons on the main thread (Chroma client + embedding
        # model + LLM) so workers don't race to build them.
        from phase_b_query.retriever import get_retriever

        get_retriever().invoke("warmup")
        get_llm()

        def answer_one(item) -> dict:
            i, qa = item
            return {
                "idx": i,
                "reference": qa["reference"],
                "rag": rag_answer(qa["question"]),
                "baseline": baseline_answer(qa["question"]),
            }

        with open(PREDS_OUT, "a", encoding="utf-8") as ckpt:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                futures = [pool.submit(answer_one, item) for item in todo]
                for n, fut in enumerate(as_completed(futures), start=1):
                    rec = fut.result()
                    ckpt.write(json.dumps(rec) + "\n")
                    ckpt.flush()  # persist progress so a kill is resumable
                    done[rec["idx"]] = rec
                    if n % 10 == 0 or n == len(todo):
                        print(f"  answered {n}/{len(todo)} (total cached {len(done)})")

    # Assemble in QA order and score.
    ordered = [done[i] for i in range(len(pairs)) if i in done]
    references = [r["reference"] for r in ordered]
    rag_preds = [r["rag"] for r in ordered]
    base_preds = [r["baseline"] for r in ordered]

    print(f"\nScoring {len(ordered)} pairs (first BERTScore call downloads roberta-large) ...")
    results = {
        "n": len(ordered),
        "rag": metrics.evaluate(rag_preds, references),
        "no_rag": metrics.evaluate(base_preds, references),
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
    print(f"\nBERTScore target {r['target_bertscore']}: {hit}  (n={r['n']})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run RAG vs no-RAG evaluation")
    ap.add_argument("--limit", type=int, default=240, help="QA pairs to score")
    args = ap.parse_args()
    run(args.limit)


if __name__ == "__main__":
    main()
