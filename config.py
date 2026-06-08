"""Central configuration — the paper's exact settings live here.

These are constants only (no logic) so every module shares one source of
truth. Adjust here, not in the pipeline code.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_RAW = ROOT / "data" / "raw"            # drop the book PDF here
DATA_PROCESSED = ROOT / "data" / "processed"  # cleaned text, chunk dumps
CHROMA_DIR = ROOT / "chroma_db"             # persisted vector store

# ── Phase A: chunking (paper's exact settings) ───────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ── Phase A: embeddings ──────────────────────────────────────────────────
# Lead candidate; alternatives: intfloat/e5-large-v2, all-MiniLM-L6-v2.
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# ── Phase B: retrieval (paper's settings) ────────────────────────────────
SEARCH_TYPE = "mmr"   # Maximum Marginal Relevance
TOP_K = 10            # chunks returned per question
MMR_LAMBDA = 0.5      # relevance vs. diversity balance knob

# ── Phase B: LLM (swap freely; compare per the plan) ─────────────────────
LLM_PROVIDER = "deepseek"   # deepseek | openai | anthropic | qwen | ollama
LLM_MODEL = "deepseek-chat"

# ── Evaluation ───────────────────────────────────────────────────────────
BERTSCORE_TARGET = 0.80
