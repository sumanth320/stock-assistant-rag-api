# LLMOps Stock Assistant

An AI-powered stock assistant built with FastAPI that supports:

- **Intent routing** (`tool`, `rag`, `direct`, `clarify`)
- **Live stock quote retrieval** using **Finnhub API**
- **RAG (Retrieval-Augmented Generation)** over custom trading education documents
- **LLM response generation** via **Ollama** (`llama3:latest`)

---

## Features

- Smart query routing based on semantic similarity (`sentence-transformers`)
- Dynamic route handling:
  - **tool** → real-time stock data from Finnhub
  - **rag** → answers grounded in indexed local documents (ChromaDB)
  - **direct** → general LLM response
  - **clarify** → asks follow-up when input is ambiguous
- Local vector database with **ChromaDB**
- Simple API interface through a single endpoint: `POST /ask`

---

## Project Structure


app/
  config.py
  index_documents.py
  llm_client.py
  main.py
  orchestrator.py
  prefilter.py
  prompt_builder.py
  rag.py
  requirements.txt
  tools.py


## How It Works
Client sends a question to POST /ask
route_query() in app/orchestrator.py decides route:
tool, rag, direct, or clarify
build_prompt() in app/prompt_builder.py creates route-specific prompt
generate_response() in app/llm_client.py calls Ollama
API returns final answer + selected route (except clarify response)
 
## Prerequisites
Python 3.10+ (recommended)
Ollama installed
llama3:latest model available in Ollama
Finnhub API key
 
Setup
1) Clone and enter project
   
git clone <your-repo-url>
cd llmops-assistant

2) Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

3) Install dependencies
pip install -r app/requirements.txt

4) Create .env file at project root
cat > .env << 'EOF'
FINNHUB_API_KEY=your_finnhub_api_key_here
EOF
Optional: If you see Hugging Face unauthenticated warnings while downloading embedding models, you can set HF_TOKEN, but it is not required for normal local use.
 
## Index RAG Documents (one-time or when docs change)
Run the indexing script to populate local ChromaDB:
python app/index_documents.py
This creates/updates local vector data in chroma_db/.
 
## Run the Application
1) Start Ollama (if not already running)
ollama serve
If you get address already in use, Ollama is already running.
2) Ensure model exists
ollama list
If missing:
ollama pull llama3
3) Start FastAPI server
uvicorn app.main:app --reload
Server runs on:
http://127.0.0.1:8000
Swagger docs: http://127.0.0.1:8000/docs
 
API Usage
Endpoint
POST /ask
Request body
{
  "question": "What is AAPL stock price?",
  "user_id": "u1",
  "session_id": "s1"
}
Example curl
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is AAPL stock price?","user_id":"u1","session_id":"s1"}'
Possible response patterns
tool/direct/rag route
{
  "route": "tool",
  "answer": "According to the supplied market data..."
}
clarify route
{
  "message": "Could you specify the stock symbol (e.g. AAPL)?"
}
 

## Configuration Notes
app/config.py controls RAG thresholds and retrieval behavior
app/prefilter.py defines route example phrases used for semantic routing
app/tools.py handles ticker extraction + Finnhub calls
app/rag.py handles retrieval from ChromaDB
 
## Troubleshooting
ModuleNotFoundError: chromadb
Install dependencies in your active virtual environment:
pip install -r app/requirements.txt

Finnhub returns invalid key (401)
Check .env:
FINNHUB_API_KEY=your_real_key
No quotes, no spaces around =.

Stock response shows 0
Usually caused by invalid key, rate limiting, or request failure from provider. Verify key and test endpoint manually.
 
## Roadmap Ideas
Add conversation memory using session_id
Add robust error handling for provider failures/rate limits
Improve ticker/entity extraction
Add tests for route selection and tool integration
Add Docker support and CI
 
## License
Add your preferred license (MIT/Apache-2.0/etc.).
















