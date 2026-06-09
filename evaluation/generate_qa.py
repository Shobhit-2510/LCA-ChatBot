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
import math
import re
from collections import defaultdict

from pydantic import BaseModel, Field

import config
from phase_b_query.llm import get_llm

# Pattern to identify numbered chapters like "8 Scope Definition" (for balanced sampling)
_NUMBERED = re.compile(r"^\d+\s")

CHUNKS_IN = config.DATA_PROCESSED / "chunks.jsonl"  # Input chunks from Phase A
QA_OUT = config.DATA_PROCESSED / "qa_pairs.jsonl"  # Output QA pairs

MIN_CHARS = 400  # Skip very short chunks (can't make good questions)
SKIP_CHAPTERS = {"Front Matter", "Unknown"}  # Skip front/back matter


# Single Q&A pair (for Claude structured output)
class QAPair(BaseModel):
    question: str = Field(description="A self-contained question answerable from the passage alone")
    answer: str = Field(description="A concise answer grounded only in the passage")


# Container for multiple Q&A pairs (what Claude returns)
class QASet(BaseModel):
    pairs: list[QAPair]


# Prompt template for Claude to generate Q&A pairs from a chunk
_INSTRUCTION = (
    "You are creating an exam for a life-cycle-assessment course. From the "
    "passage below (from an LCA textbook), write exactly {k} question-answer "
    "pairs. Each question must be answerable from the passage ALONE and be "
    "self-contained (no 'according to the text'). Each answer must be concise "
    "and grounded only in the passage.\n\nPassage:\n{text}"
)


def _load_chunks() -> list[dict]:
    """Load chunks from Phase A, filtering for quality."""
    if not CHUNKS_IN.exists():
        raise SystemExit(f"Run Phase A first — missing {CHUNKS_IN}")
    rows = [json.loads(l) for l in open(CHUNKS_IN, encoding="utf-8")]
    # Filter: keep only chunks that are long enough and not front/back matter
    return [
        r
        for r in rows
        if len(r["text"]) >= MIN_CHARS
        and r["metadata"].get("chapter") not in SKIP_CHAPTERS
    ]


def _sample(rows: list[dict], n: int) -> list[dict]:
    """Sample evenly spaced rows for broad chapter coverage."""
    if n >= len(rows):  # If n is large enough, return all
        return rows
    step = len(rows) / n  # Evenly space samples across the book
    return [rows[int(i * step)] for i in range(n)]


def _by_section(rows: list[dict]) -> dict[str, list[dict]]:
    """Group chunks by numbered chapter for balanced QA generation."""
    sections: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        ch = r["metadata"].get("chapter") or ""
        if _NUMBERED.match(ch):  # Only keep numbered chapters (e.g., "8 Scope Definition")
            sections[ch].append(r)
    return dict(sections)


def _gen_from_chunk(llm, row: dict, k: int) -> list[dict]:
    """Generate k pairs from one chunk; return [] on failure."""
    meta = row["metadata"]
    prompt = _INSTRUCTION.format(k=k, text=row["text"])  # Fill prompt template
    try:
        result: QASet = llm.invoke(prompt)  # Call Claude with structured output
    except Exception as e:
        print(f"      skipped a chunk ({type(e).__name__})")
        return []  # Skip this chunk if generation fails
    # Format pairs with metadata for saving
    return [
        {
            "question": p.question,
            "reference": p.answer,  # Reference answer from Claude
            "chapter": meta.get("chapter"),
            "pdf_page": meta.get("pdf_page"),
        }
        for p in result.pairs
    ]


def generate_balanced(per_section: int, pairs_per_chunk: int) -> list[dict]:
    """Generate exactly per_section pairs for every numbered chapter."""
    sections = _by_section(_load_chunks())  # Group chunks by numbered chapter
    n = len(sections)
    chunks_per_section = math.ceil(per_section / pairs_per_chunk)  # Chunks needed per chapter
    print(
        f"Balanced: {per_section} pairs x {n} sections = {per_section * n} target "
        f"({chunks_per_section} chunks/section x {pairs_per_chunk} pairs)"
    )

    llm = get_llm().with_structured_output(QASet)  # LLM with structured output
    out: list[dict] = []
    # Process each numbered chapter in order
    for i, ch in enumerate(sorted(sections, key=lambda c: int(c.split()[0])), start=1):
        chunks = _sample(sections[ch], chunks_per_section)  # Sample chunks evenly
        pairs: list[dict] = []
        for row in chunks:
            pairs.extend(_gen_from_chunk(llm, row, pairs_per_chunk))  # Generate pairs
            if len(pairs) >= per_section:  # Stop if we have enough pairs
                break
        pairs = pairs[:per_section]  # Keep exactly per_section pairs
        out.extend(pairs)
        print(f"  [{i}/{n}] {ch[:40]:<40} {len(pairs)} pairs  (total {len(out)})")

    _save(out)
    return out


def _save(rows: list[dict]) -> None:
    """Save QA pairs to JSONL file."""
    QA_OUT.parent.mkdir(parents=True, exist_ok=True)  # Create output directory
    with open(QA_OUT, "w", encoding="utf-8") as f:
        for r in rows:  # Write each pair as a JSON line
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(rows)} pairs -> {QA_OUT}")


def generate(num_chunks: int, pairs_per_chunk: int) -> list[dict]:
    """Generate QA pairs from evenly-sampled chunks (unbalanced mode)."""
    rows = _sample(_load_chunks(), num_chunks)  # Sample chunks evenly across book
    llm = get_llm().with_structured_output(QASet)  # LLM with structured output
    print(f"Generating from {len(rows)} chunks x {pairs_per_chunk} pairs ...")

    out: list[dict] = []
    for i, row in enumerate(rows, start=1):
        meta = row["metadata"]
        prompt = _INSTRUCTION.format(k=pairs_per_chunk, text=row["text"])
        try:
            result: QASet = llm.invoke(prompt)  # Generate pairs from this chunk
        except Exception as e:  # Skip chunk on error, don't fail entire run
            print(f"  [{i}/{len(rows)}] skipped ({type(e).__name__})")
            continue
        # Add each pair with metadata
        for p in result.pairs:
            out.append({
                "question": p.question,
                "reference": p.answer,
                "chapter": meta.get("chapter"),
                "pdf_page": meta.get("pdf_page"),
            })
        print(f"  [{i}/{len(rows)}] {len(result.pairs)} pairs  (total {len(out)})")

    _save(out)
    return out


def main() -> None:
    """CLI entry point with --per-section (balanced) or --num-chunks (unbalanced)."""
    ap = argparse.ArgumentParser(description="Generate the QA test set")
    ap.add_argument(
        "--per-section",
        type=int,
        help="balanced mode: generate exactly N pairs per numbered chapter",
    )
    ap.add_argument("--num-chunks", type=int, default=20, help="(unbalanced) chunks to sample")
    ap.add_argument("--pairs-per-chunk", type=int, default=5, help="Q&A pairs per chunk")
    args = ap.parse_args()

    # Use balanced mode if --per-section is specified, otherwise unbalanced
    if args.per_section:
        generate_balanced(args.per_section, args.pairs_per_chunk)
    else:
        generate(args.num_chunks, args.pairs_per_chunk)


if __name__ == "__main__":
    main()
