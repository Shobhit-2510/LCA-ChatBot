# LCA ChatBot

A Retrieval-Augmented Generation (RAG) chatbot powered by Claude for answering questions about Life Cycle Assessment (LCA) concepts and methodologies, grounded in the Hauschild et al. textbook "LCA: Theory and Practice".

## Overview

LCA ChatBot provides an intelligent conversational interface for exploring LCA concepts. The system combines vector embeddings for semantic search with Claude AI for context-aware answers, ensuring responses are grounded in the source material.

Key capabilities:
- Chat interface with natural language questions
- Source citations showing chapter and page references
- Conversation history for contextual follow-up questions
- RAG pipeline combining retrieval and generation
- MMR (Maximum Marginal Relevance) for diverse, relevant results

## Architecture

The system operates in two distinct phases:

### Phase A: Ingestion (Offline)

```
PDF (Textbook) 
    -> Extract text
    -> Clean and chunk (overlapping segments)
    -> Embed with bge-large-en-v1.5
    -> Store in Chroma vector database
```

Runs once to build the knowledge base. Output is persisted for Phase B.

### Phase B: Query (Online)

```
User Question
    -> Retrieve top-10 MMR chunks
    -> Format context with source tags
    -> Build prompt (system + context + question)
    -> Call Claude API
    -> Extract answer and sources
    -> Display results
```

Runs for each user query. Responses include citations.

## Requirements

- Python 3.10 or higher
- Virtual environment (recommended)
- Anthropic API key (https://console.anthropic.com/)
- 2GB+ disk space (for embedding model)

## Installation

### 1. Set up Python environment

```bash
cd "c:\Users\user\My Project\LCA-ChatBot"
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
# or: source .venv/bin/activate  # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API key

Create `.env` file in project root:

```
ANTHROPIC_API_KEY=sk-ant-xxxxx
```

Keep this file private. Do not commit to version control (included in .gitignore).

### 4. Build vector database (first time only)

```bash
python -m phase_a_ingestion.build_index
```

This process:
- Extracts text from data/raw/Textbook.pdf
- Chunks text with configurable size and overlap
- Generates embeddings (downloads ~1.3GB model on first run)
- Stores vectors in chroma_db/

Expected time: 5-10 minutes on CPU.

### 5. Start chatbot

```bash
streamlit run app/streamlit_app.py
```

Opens at http://localhost:8501

## Usage

Type questions naturally in the chat interface. Examples:

- What is a functional unit in LCA?
- What are the main phases of LCA?
- How is environmental impact calculated?
- What is the difference between cradle-to-grave and cradle-to-cradle?

Each answer includes source citations in the format "Chapter Name, p.XX".

## Project Structure

```
LCA-ChatBot/
├── phase_a_ingestion/          Offline indexing pipeline
│   ├── build_index.py          Main ingestion script
│   ├── extract.py              PDF text extraction
│   ├── clean.py                Text cleaning
│   ├── chunk.py                Text segmentation
│   └── embed.py                Embedding model setup
├── phase_b_query/              Online query pipeline
│   ├── rag_pipeline.py         Main orchestrator
│   ├── retriever.py            Vector search
│   ├── prompt.py               Prompt template
│   └── llm.py                  LLM setup
├── app/                        User interface
│   └── streamlit_app.py        Chat UI
├── evaluation/                 System evaluation
│   ├── qa_generator.py         Generate test Q&A
│   ├── metrics.py              Evaluation metrics
│   └── compare.py              Model comparison
├── data/
│   ├── raw/                    Input data
│   └── processed/              Output artifacts
├── chroma_db/                  Vector database (generated)
├── config.py                   Global configuration
├── requirements.txt            Dependencies
├── .env                        API keys (not in git)
└── Readme.md                   This file
```

## Configuration

Edit `config.py` to customize system behavior:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| CHUNK_SIZE | 1000 | Characters per chunk |
| CHUNK_OVERLAP | 200 | Character overlap between chunks |
| EMBEDDING_MODEL | BAAI/bge-large-en-v1.5 | Embedding model identifier |
| TOP_K | 10 | Number of chunks retrieved per query |
| FETCH_K | 30 | Candidates evaluated before re-ranking |
| MMR_LAMBDA | 0.5 | Relevance vs diversity balance (0=diversity, 1=relevance) |
| LLM_PROVIDER | anthropic | LLM provider |
| LLM_MODEL | claude-opus-4-8 | Claude model version |
| LLM_MAX_TOKENS | 1024 | Maximum answer length |
| SEARCH_TYPE | mmr | Search strategy (Maximum Marginal Relevance) |

## Development

### Rebuild vector database

```bash
python -m phase_a_ingestion.build_index
```

Deletes existing database and creates fresh index. Use when changing CHUNK_SIZE or source PDF.

### Run evaluation

```bash
python -m evaluation.qa_generator
python -m evaluation.compare
```

Generates test Q&A pairs and computes evaluation metrics (BERT-score, ROUGE).

### Code quality checks

```bash
pylint phase_a_ingestion phase_b_query app
mypy --strict .
```

## How RAG Works

Retrieval-Augmented Generation combines three steps:

1. Retrieval: Vector similarity search finds passages semantically related to the question
2. Augmentation: Relevant passages are inserted into the prompt as grounding context
3. Generation: LLM reads context and generates an informed response

This approach constrains answers to the source material, reducing hallucinations and providing verifiable citations.

## Dependencies

Core dependencies:
- langchain, langchain-community, langchain-text-splitters: LLM orchestration
- chromadb, langchain-chroma: Vector database
- sentence-transformers, langchain-huggingface: Embeddings (bge-large-en-v1.5)
- langchain-anthropic: Claude API integration
- streamlit: Web UI
- pymupdf, pdfplumber: PDF extraction

Evaluation:
- bert-score, rouge-score: Evaluation metrics
- scikit-learn: Statistical analysis
- matplotlib: Visualization

Utilities:
- python-dotenv: Environment variable loading
- tqdm: Progress bars

See requirements.txt for complete list. Versions are unpinned; freeze with `pip freeze > requirements.lock.txt` after validation.

## Citation

This project builds on the Hauschild et al. textbook:

```
Hauschild, Michael Z., Ralph K. Rosenbaum, and Stig Irving Olsen. 
"LCA: Theory and Practice." Springer, 2018.
```

## Troubleshooting

**ModuleNotFoundError when running scripts**
- Verify virtual environment is activated
- Run from project root directory
- Check sys.path.insert in app/streamlit_app.py

**ANTHROPIC_API_KEY not found**
- Create .env file in project root
- Add ANTHROPIC_API_KEY=sk-ant-xxxxx
- Ensure .env is in .gitignore

**Slow first run**
- Embedding model downloads on first use (~1.3GB)
- CPU inference is intentionally slow
- Model is cached after first run

**Vector database errors**
- Delete chroma_db/ directory
- Rebuild with: python -m phase_a_ingestion.build_index

**Streamlit not found**
- Run: pip install -r requirements.txt
- Verify requirements.txt is current

## License

This project uses copyrighted material (Hauschild et al. textbook) for educational purposes. Refer to the textbook's usage terms.

## Support

For issues:
1. Review troubleshooting section above
2. Check config.py settings match your environment
3. Verify all dependencies in requirements.txt are installed
4. Ensure .env contains valid ANTHROPIC_API_KEY
