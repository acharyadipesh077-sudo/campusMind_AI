# CAMPUSMIND AI

Nepali Agentic RAG chatbot for academic and mental health support. The system runs locally with Ollama, FAISS, BM25, LangGraph, FastAPI, Streamlit, and SQLite.

## Target Hardware

- Lenovo LOQ class laptop
- 24 GB RAM
- NVIDIA RTX 4050 6 GB VRAM
- Ollama model: `qwen2.5:7b-instruct` or `qwen2.5:7b`
- Embedding model: `BAAI/bge-m3`

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
ollama pull qwen2.5:7b-instruct
```

If Ollama uses the shorter model name on your machine, set `OLLAMA_MODEL=qwen2.5:7b` in `.env`.

## Dataset

Expected Excel columns:

- `S.N.`
- `question`
- `context`
- `answer`
- `domain`

Place the file anywhere and point `DATASET_PATH` to it in `.env`, or pass it directly:

```powershell
python -m scripts.build_index --dataset "D:\campusMind_AI\data\campusmind_dataset.xlsx"
```

The index stores embeddings for `question + context + answer`, using normalized vectors.

## Run

Start API:

```powershell
uvicorn campusmind.api:app --reload --host 127.0.0.1 --port 8000
```

Start the production ChatGPT-like streaming UI in a second terminal:

```powershell
streamlit run ui/chatgpt_app.py
```

The older demo UI remains available at `ui/streamlit_app.py`.

## Project Layout

```text
campusMind_AI/
  campusmind/
    agents.py          LangGraph agent pipeline
    api.py             FastAPI app
    auth.py            bcrypt + JWT helpers
    config.py          settings
    database.py        SQLite schema and operations
    evaluation.py      BLEU, METEOR, precision, recall, F1, consistency
    llm.py             Ollama client
    retrieval.py       FAISS + BM25 hybrid retrieval
    schemas.py         Pydantic request/response models
    text.py            Nepali normalization and emergency detection
  scripts/
    build_index.py     Excel ingestion and index builder
  ui/
    streamlit_app.py   ChatGPT-style Streamlit frontend
```

## Mental Health Safety

Emergency self-harm or suicide intent bypasses retrieval and returns an immediate supportive crisis response in Nepali. This is not a replacement for professional care.
