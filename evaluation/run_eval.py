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

MAX_WORKERS = 6  # Number of parallel workers for answering
RATE_PER_MIN = 45  # Requests per minute limit (org cap is 50 for Opus)

# System message for no-RAG baseline (Claude answering from its own knowledge)
_BASELINE_SYSTEM = (
    "You are an LCA expert assistant. Answer the question concisely from your "
    "own knowledge."
)


class RateLimiter:
    """Thread-safe sliding-window rate limiter (requests per minute)."""

    def __init__(self, rate_per_min: int):
        self.rate = rate_per_min
        self.calls: deque[float] = deque()  # Timestamps of recent calls
        self.lock = threading.Lock()

    def acquire(self) -> None:
        """Block until we're under the rate limit, then record a call."""
        while True:
            with self.lock:
                now = time.monotonic()
                # Remove timestamps older than 60 seconds
                while self.calls and now - self.calls[0] >= 60:
                    self.calls.popleft()
                # If we're under the limit, record this call and return
                if len(self.calls) < self.rate:
                    self.calls.append(now)
                    return
                # Otherwise, calculate how long to wait
                wait = 60 - (now - self.calls[0])
            time.sleep(max(wait, 0.05))  # Sleep then retry


_limiter = RateLimiter(RATE_PER_MIN)  # Shared rate limiter for all threads


def baseline_answer(question: str) -> str:
    """Answer without retrieval (no-RAG baseline for comparison)."""
    _limiter.acquire()  # Wait for rate limit slot
    resp = get_llm().invoke([("system", _BASELINE_SYSTEM), ("human", question)])
    return _text(resp.content)


def rag_answer(question: str) -> str:
    """Answer with retrieval-augmented generation (RAG)."""
    _limiter.acquire()  # Wait for rate limit slot
    return answer(question)["answer"]  # Use RAG pipeline


def _load_checkpoint() -> dict[int, dict]:
    """Load previously completed predictions to avoid re-answering."""
    if not PREDS_OUT.exists():
        return {}
    done = {}
    for line in open(PREDS_OUT, encoding="utf-8"):
        rec = json.loads(line)
        done[rec["idx"]] = rec  # Map index to prediction record
    return done


def run(limit: int) -> dict:
    """Evaluate RAG vs baseline on QA pairs: answer both, score both, show results."""
    if not QA_IN.exists():
        raise SystemExit(f"Run evaluation.generate_qa first — missing {QA_IN}")
    pairs = [json.loads(l) for l in open(QA_IN, encoding="utf-8")][:limit]  # Load QA pairs

    done = _load_checkpoint()  # Load cached predictions
    todo = [(i, qa) for i, qa in enumerate(pairs) if i not in done]  # Find what's left
    print(
        f"Evaluating {len(pairs)} QA pairs (RAG vs no-RAG) — "
        f"{len(done)} cached, {len(todo)} to answer, {MAX_WORKERS}-way / {RATE_PER_MIN} rpm"
    )

    if todo:
        # Warm up singletons on main thread so workers don't race to initialize
        from phase_b_query.retriever import get_retriever

        get_retriever().invoke("warmup")  # Initialize Chroma client and embeddings
        get_llm()  # Initialize LLM

        def answer_one(item) -> dict:
            """Answer one QA pair with both RAG and baseline."""
            i, qa = item
            return {
                "idx": i,
                "reference": qa["reference"],
                "rag": rag_answer(qa["question"]),  # RAG answer
                "baseline": baseline_answer(qa["question"]),  # Baseline answer
            }

        # Answer all pairs in parallel with checkpointing
        with open(PREDS_OUT, "a", encoding="utf-8") as ckpt:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                futures = [pool.submit(answer_one, item) for item in todo]
                for n, fut in enumerate(as_completed(futures), start=1):
                    rec = fut.result()
                    ckpt.write(json.dumps(rec) + "\n")
                    ckpt.flush()  # Persist after each answer (resumable)
                    done[rec["idx"]] = rec
                    if n % 10 == 0 or n == len(todo):
                        print(f"  answered {n}/{len(todo)} (total cached {len(done)})")

    # Assemble results in order and score all
    ordered = [done[i] for i in range(len(pairs)) if i in done]
    references = [r["reference"] for r in ordered]  # Ground truth answers
    rag_preds = [r["rag"] for r in ordered]  # RAG-generated answers
    base_preds = [r["baseline"] for r in ordered]  # Baseline answers

    print(f"\nScoring {len(ordered)} pairs (first BERTScore call downloads roberta-large) ...")
    # Score both RAG and baseline with all three metrics
    results = {
        "n": len(ordered),
        "rag": metrics.evaluate(rag_preds, references),  # RAG scores
        "no_rag": metrics.evaluate(base_preds, references),  # Baseline scores
        "target_bertscore": config.BERTSCORE_TARGET,
    }

    _print_table(results)  # Display results
    RESULTS_OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")  # Save results
    print(f"\nSaved -> {RESULTS_OUT}")
    return results


def _print_table(r: dict) -> None:
    """Print metrics comparison table (RAG vs baseline)."""
    print(f"\n{'metric':<14}{'RAG':>10}{'no-RAG':>10}{'gap':>10}")
    print("-" * 44)
    # Show each metric with RAG score, baseline score, and gap
    for k in ("bertscore_f1", "cosine", "rouge_l"):
        gap = r["rag"][k] - r["no_rag"][k]
        print(f"{k:<14}{r['rag'][k]:>10.3f}{r['no_rag'][k]:>10.3f}{gap:>+10.3f}")
    # Show if we hit the BERTScore target
    hit = "PASS" if r["rag"]["bertscore_f1"] >= r["target_bertscore"] else "below"
    print(f"\nBERTScore target {r['target_bertscore']}: {hit}  (n={r['n']})")


def main() -> None:
    """CLI entry point for evaluation runner."""
    ap = argparse.ArgumentParser(description="Run RAG vs no-RAG evaluation")
    ap.add_argument("--limit", type=int, default=240, help="QA pairs to score")
    args = ap.parse_args()
    run(args.limit)  # Run evaluation on specified number of pairs


if __name__ == "__main__":
    main()
