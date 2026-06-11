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
import random  # Optional: for random sampling instead of evenly spaced
import re
from collections import defaultdict

from pydantic import BaseModel, Field

import config
from phase_b_query.llm import get_llm

# Pattern to identify numbered chapters like "8 Scope Definition" (for balanced sampling)
_NUMBERED = re.compile(r"^\d+\s")

CHUNKS_IN = config.DATA_PROCESSED / "chunks.jsonl"  # Input chunks of book from Phase A was created when building the vector database
QA_OUT = config.DATA_PROCESSED / "qa_pairs.jsonl"  # Output QA pairs

MIN_CHARS = 400  # Skip very short chunks because they may not have enough content for good Q&A generation
SKIP_CHAPTERS = {"Front Matter", "Unknown"}  # Skip front/back matter and untagged chunks to focus on main content chapters


# Single Q&A pair (for Claude structured output)
class QAPair(BaseModel):
    # When someone creates a QAPair object, they MUST provide a 'question' that is a "string" and Field helps in describing what that question should be like.
    question: str = Field(description="A self-contained question answerable from the passage alone") # The description helps Claude understand what you want in the question
    answer: str = Field(description="A concise answer grounded only in the passage") # The description helps Claude understand what you want in the answer


# Container for multiple Q&A pairs (what Claude returns)
class QASet(BaseModel):
    # When someone creates a QASet object, they MUST provide a 'pairs' that is a "list of QAPair objects".
    pairs: list[QAPair]


# Prompt template for Claude to generate Q&A pairs from a chunk
_INSTRUCTION = (
    "You are creating an exam for a life-cycle-assessment course. From the "
    "passage below (from an LCA textbook), write exactly {k} question-answer "
    "pairs. Each question must be answerable from the passage ALONE and be "
    "self-contained (no 'according to the text'). Each answer must be concise "
    "and grounded only in the passage.\n\nPassage:\n{text}"
) # k is a placeholder for string which will be replaced by the actual number of pairs when we use _INSTRUCTION.format(k=..., text=...) to create the final prompt for Claude.

# In short, this function is used to load and filter book chunks from Phase A.
def _load_chunks() -> list[dict]:
    """Load chunks from Phase A, filtering for quality."""
    if not CHUNKS_IN.exists():  # Check if chunks.jsonl exists (Phase A output)
        raise SystemExit(f"Run Phase A first — missing {CHUNKS_IN}")
    rows = [json.loads(l) for l in open(CHUNKS_IN, encoding="utf-8")]  # Parse each line as JSON
    return [
        r
        for r in rows
        if len(r["text"]) >= MIN_CHARS  # Keep only chunks with 400+ characters
        and r["metadata"].get("chapter") not in SKIP_CHAPTERS  # Exclude front/back matter
    ]

# This function samples n rows evenly from the input list of rows to ensure broad coverage across the book. If n is larger than the number of rows, it returns all rows.
# Used when we want to sample a specific number of chunks from each chapter for balanced QA generation, or when we want to sample a specific number of chunks across the whole book for unbalanced generation.
"""Here is a problem it will generate same chunks if asked to generate 250 QA pairs 2 times.
"""
def _sample(rows: list[dict], n: int) -> list[dict]:
    """Sample evenly spaced rows for broad chapter coverage."""
    if n >= len(rows):  # If n is large enough, return all
        return rows
    step = len(rows) / n  # Evenly space samples across the book
    random.shuffle(rows)  # Randomize order could be helpful if random QA generation is desired
    return [rows[int(i * step)] for i in range(n)]

# Input looks like: [{"text": "...", "metadata": {"chapter": "8 Scope Definition"}}, {"text": "...", "metadata": {"chapter": "9 Methodology"}}, ...]
# Output is a dict grouping rows by chapter: {"8 Scope Definition": [{"text": "...", "metadata": {...}}, ...], "9 Methodology": [{"text": "...", "metadata": {...}}, ...], ...}
def _by_section(rows: list[dict]) -> dict[str, list[dict]]:
    """Group chunks by numbered chapter for balanced QA generation."""
    sections: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        ch = r["metadata"].get("chapter") or ""
        if _NUMBERED.match(ch):  # Only keep numbered chapters (e.g., "8 Scope Definition") because we want to balance by main content chapters, not front/back matter
            sections[ch].append(r)
    return dict(sections)


def _gen_from_chunk(llm, row: dict, k: int) -> list[dict]:
    """Generate k question-answer pairs from a single text chunk using Claude.

    Takes a chunk of text from Phase A (with metadata) and asks Claude to generate
    k self-contained Q&A pairs. Each pair includes the source chapter and page.
    If Claude fails (API error, timeout, etc.), returns empty list to allow graceful
    continuation instead of crashing the entire evaluation.

    Args:
        llm: Claude LLM with structured output enabled (returns QASet objects)
        row: Chunk dict with keys: {"text": str, "metadata": {chapter, pdf_page, ...}}
        k: Number of Q&A pairs to generate from this chunk

    Returns:
        List of Q&A pair dicts with format:
        [{"question": str, "reference": str, "chapter": str, "pdf_page": int}, ...]
        Empty list [] if generation fails.
    """
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
    """Generate balanced QA pairs across all numbered chapters.

    Distributes QA pair generation uniformly across chapters by sampling chunks evenly
    from each chapter and generating pairs until the target per-chapter quota is met.
    Ensures consistent representation across the document structure.

    Args:
        per_section: Target number of QA pairs to generate per chapter.
        pairs_per_chunk: Number of QA pairs to generate from each chunk.

    Returns:
        List of QA pair dictionaries. Pairs are also persisted to JSONL file.
    """
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
    """Generate QA pairs from globally sampled chunks without chapter-level distribution control.

    Samples chunks uniformly across the entire document and generates pairs from each chunk.
    Unlike generate_balanced(), this approach does not guarantee equal representation across
    chapters—chapters with more content may contribute more pairs than others. Useful for
    generating a fixed total number of pairs (approximately num_chunks * pairs_per_chunk)
    without chapter-level constraints.

    Args:
        num_chunks: Total number of chunks to sample from the entire document.
        pairs_per_chunk: Number of QA pairs to generate from each chunk.

    Returns:
        List of QA pair dictionaries. Pairs are also persisted to JSONL file.
    """
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

    # Use balanced mode if --per-section is specified(means default=10 is given), otherwise unbalanced
    # here it is running as unbalanced 
    if args.per_section:
        generate_balanced(args.per_section, args.pairs_per_chunk)
    else:
        generate(args.num_chunks, args.pairs_per_chunk)


if __name__ == "__main__":
    main()
