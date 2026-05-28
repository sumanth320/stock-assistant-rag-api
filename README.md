# LLMOps Stock Assistant

An AI-powered stock assistant built with FastAPI that supports:

- **Intent routing** (`tool`, `rag`, `direct`, `clarify`)
- **Live stock quote retrieval** via **Finnhub API**
- **RAG (Retrieval-Augmented Generation)** over custom trading education documents
- **LLM response generation** using **Ollama** (`llama3:latest`)
- **Optional routing debug metadata** for route decisions

---

## Features

- Semantic routing with `sentence-transformers`
- Dynamic route handling:
  - **tool** -> real-time stock data from Finnhub
  - **rag** -> answers grounded in indexed local documents (ChromaDB)
  - **direct** -> general LLM response
  - **clarify** -> asks follow-up when input is ambiguous
- Local vector database with **ChromaDB**
- API endpoint: `POST /ask`
- Streamlit UI for quick testing

---

## Project Structure

```text
stock-assistant-rag-api/
  app/
    config.py
    index_documents.py
    llm_client.py
    logger.py
    main.py
    orchestrator.py
    prefilter.py
    prompt_builder.py
    rag.py
    requirements.txt
    tools.py
  ui/
    streamlit_app.py
  chroma_db/
```

---

## End-to-End Flow

1. Client sends a question to `POST /ask`
2. `route_query()` in `app/orchestrator.py` selects route (`tool`, `rag`, `direct`, `clarify`)
3. `build_prompt()` in `app/prompt_builder.py` creates route-specific prompt
4. `generate_response()` in `app/llm_client.py` calls Ollama
5. API returns final answer or clarify message

---

## Prerequisites

- Python 3.10+ (recommended)
- Ollama installed
- `llama3:latest` model available in Ollama
- Finnhub API key

---

## Setup

### 1) Clone and enter project

```bash
git clone <your-repo-url>
cd stock-assistant-rag-api
```

### 2) Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r app/requirements.txt
```

### 4) Create `.env` file at project root

```bash
cat > .env << 'EOF'
FINNHUB_API_KEY=<your-valid-key>
EOF
```

Optional: If you see Hugging Face unauthenticated warnings while downloading embedding models, you can set `HF_TOKEN`.

---

## Index RAG Documents (one-time or when docs change)

```bash
python app/index_documents.py
```

This creates/updates local vector data in `chroma_db/`.

---

## Run the Application

### 1) Start Ollama (if not already running)

```bash
ollama serve
```

If you get `address already in use`, Ollama is already running.

### 2) Ensure model exists

```bash
ollama list
```

If missing:

```bash
ollama pull llama3
```

### 3) Start FastAPI server

```bash
uvicorn app.main:app --reload
```

Server runs on:

- `http://127.0.0.1:8000`
- Swagger docs: `http://127.0.0.1:8000/docs`

### 4) Start Streamlit UI (new terminal)

```bash
streamlit run ui/streamlit_app.py
```

---

## API Usage

### Endpoint

`POST /ask`

### Request body

```json
{
  "question": "What is AAPL stock price?",
  "user_id": "demo-user",
  "session_id": "session-123"
}
```

### Example curl

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is AAPL stock price?","user_id":"demo-user","session_id":"session-123"}'
```

### Possible response patterns

#### `tool` / `direct` / `rag` route

```json
{
  "route": "tool",
  "answer": "According to the supplied market data...",
  "debug": {
    "best_route": "tool",
    "best_score": 0.58
  }
}
```

#### `clarify` route

```json
{
  "message": "Could you specify the stock symbol (e.g. AAPL)?",
  "debug": {
    "decision": "clarify_low_tool_confidence"
  }
}
```

`debug` is optional and may not be present in all responses.

---

## Configuration Notes

- `app/config.py` controls confidence thresholds and RAG retrieval behavior
- `HIGH_CONFIDENCE` is currently `0.50`
- `app/prefilter.py` defines route example phrases used for semantic routing
- `app/tools.py` handles ticker extraction + Finnhub calls
- `app/rag.py` handles retrieval from ChromaDB
- `app/logger.py` prints JSON trace logs in the FastAPI terminal

---

## Troubleshooting

### `Address already in use` on FastAPI startup

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
kill -15 <PID>
```

If needed, force stop:

```bash
kill -9 <PID>
```

Or use another port:

```bash
uvicorn app.main:app --reload --port 8001
```

### Finnhub key not loaded

```bash
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('FINNHUB_API_KEY set:', bool(os.getenv('FINNHUB_API_KEY')))"
```

If output is `False`, ensure `.env` is in project root and restart FastAPI.

### Stock response shows `0`

Common causes: missing/invalid API key, request failure, or provider/rate-limit issues.

Test provider directly:

```bash
curl -s "https://finnhub.io/api/v1/quote?symbol=AAPL&token=$FINNHUB_API_KEY"
```

### `requests.exceptions.JSONDecodeError` in Streamlit

This usually means backend returned non-JSON (often HTTP 500). Check FastAPI terminal logs for traceback.

---

## Roadmap Ideas

- Add conversation memory using `session_id`
- Add tests for route selection and tool integration
- Improve ticker/entity extraction
- Add robust provider error handling/rate-limit diagnostics
- Add Docker support and CI

---

## License

Add your preferred license (MIT / Apache-2.0 / etc.).
