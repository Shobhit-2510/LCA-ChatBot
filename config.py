"""Central configuration — the paper's exact settings live here.

These are constants only (no logic) so every module shares one source of
truth. Adjust here, not in the pipeline code.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent  # Project root directory
DATA_RAW = ROOT / "data" / "raw"  # Where to put the book PDF
DATA_PROCESSED = ROOT / "data" / "processed"  # Cleaned text and chunks output
CHROMA_DIR = ROOT / "chroma_db"  # Persisted vector database for retrieval

# ── Phase A: chunking (paper's exact settings) ───────────────────────────
CHUNK_SIZE = 1000  # Text chunk size in characters
CHUNK_OVERLAP = 200  # Overlap between chunks for continuity

# ── Phase A: embeddings ──────────────────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"  # Model to convert text to vectors

# ── Phase B: retrieval (paper's settings) ────────────────────────────────
SEARCH_TYPE = "mmr"  # Maximum Marginal Relevance (balance relevance + diversity)
TOP_K = 10  # Number of chunks to retrieve per question
MMR_LAMBDA = 0.5  # Weight: 0=diversity, 1=relevance, 0.5=balanced

# ── Phase B: LLM (swap freely; compare per the plan) ─────────────────────
LLM_PROVIDER = "anthropic"  # LLM service provider
LLM_MODEL = "claude-opus-4-8"  # Specific model
LLM_MAX_TOKENS = 1024  # Max tokens per response

# ── Evaluation ───────────────────────────────────────────────────────────
BERTSCORE_TARGET = 0.80  # Target BERTScore F1 from paper
