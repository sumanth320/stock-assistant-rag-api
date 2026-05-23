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

```text
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
