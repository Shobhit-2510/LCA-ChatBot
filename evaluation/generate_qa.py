"""Build the QA test set (Evaluation Part 1).

Feeds book chunks to Claude ("write N Q&A pairs from this text") and saves
self-contained question/reference-answer pairs tagged by chapter/page. The
paper aimed for 500-1,000 qualified pairs; defaults here are modest to keep
API cost low — scale up with the CLI flags.

Usage:
    python -m evaluation.generate_qa --num-chunks 50 --pairs-per-chunk 10

Output: data/processed/qa_pairs.jsonl
    {"question", "reference", "chapter", "pdf_page"}
"""

from __future__ import annotations

import argparse
import json

from pydantic import BaseModel, Field

import config
from phase_b_query.llm import get_llm

CHUNKS_IN = config.DATA_PROCESSED / "chunks.jsonl"
QA_OUT = config.DATA_PROCESSED / "qa_pairs.jsonl"

# Skip chunks too short to yield a meaningful question.
MIN_CHARS = 400
# Skip front/back matter (title page, contents, index) — not real content.
SKIP_CHAPTERS = {"Front Matter", "Unknown"}


class QAPair(BaseModel):
    question: str = Field(description="A self-contained question answerable from the passage alone")
    answer: str = Field(description="A concise answer grounded only in the passage")


class QASet(BaseModel):
    pairs: list[QAPair]


_INSTRUCTION = (
    "You are creating an exam for a life-cycle-assessment course. From the "
    "passage below (from an LCA textbook), write exactly {k} question-answer "
    "pairs. Each question must be answerable from the passage ALONE and be "
    "self-contained (no 'according to the text'). Each answer must be concise "
    "and grounded only in the passage.\n\nPassage:\n{text}"
)


def _load_chunks() -> list[dict]:
    if not CHUNKS_IN.exists():
        raise SystemExit(f"Run Phase A first — missing {CHUNKS_IN}")
    rows = [json.loads(l) for l in open(CHUNKS_IN, encoding="utf-8")]
    return [
        r
        for r in rows
        if len(r["text"]) >= MIN_CHARS
        and r["metadata"].get("chapter") not in SKIP_CHAPTERS
    ]


def _sample(rows: list[dict], n: int) -> list[dict]:
    """Evenly spaced sample across the book for broad chapter coverage."""
    if n >= len(rows):
        return rows
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n)]


def generate(num_chunks: int, pairs_per_chunk: int) -> list[dict]:
    rows = _sample(_load_chunks(), num_chunks)
    llm = get_llm().with_structured_output(QASet)
    print(f"Generating from {len(rows)} chunks x {pairs_per_chunk} pairs ...")

    out: list[dict] = []
    for i, row in enumerate(rows, start=1):
        meta = row["metadata"]
        prompt = _INSTRUCTION.format(k=pairs_per_chunk, text=row["text"])
        try:
            result: QASet = llm.invoke(prompt)
        except Exception as e:  # one bad chunk shouldn't sink the run
            print(f"  [{i}/{len(rows)}] skipped ({type(e).__name__})")
            continue
        for p in result.pairs:
            out.append(
                {
                    "question": p.question,
                    "reference": p.answer,
                    "chapter": meta.get("chapter"),
                    "pdf_page": meta.get("pdf_page"),
                }
            )
        print(f"  [{i}/{len(rows)}] {len(result.pairs)} pairs  (total {len(out)})")

    QA_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(QA_OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(out)} pairs -> {QA_OUT}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the QA test set")
    ap.add_argument("--num-chunks", type=int, default=20, help="chunks to sample")
    ap.add_argument("--pairs-per-chunk", type=int, default=5, help="Q&A pairs each")
    args = ap.parse_args()
    generate(args.num_chunks, args.pairs_per_chunk)


if __name__ == "__main__":
    main()
