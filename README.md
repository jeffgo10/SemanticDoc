# SemanticDoc

A local, privacy-first **Multi-Document RAG (Retrieval-Augmented Generation)** platform powered by **PyTorch**, **Hugging Face Embeddings**, **FAISS**, **Ollama**, **FastAPI**, and **Nuxt 3**.

SemanticDoc lets you upload multiple PDFs (handbooks, resumes/CVs, technical manuals), automatically normalizes and chunks text, indexes embeddings into a persistent FAISS vector store, and lets you chat across all documents or filter queries to a specific PDF—with transparent source citations and L2 vector distance scores.

---

## ✨ Key Features

- **100% Local & Private RAG** — Runs embeddings on local PyTorch (`cuda`, `mps`, or `cpu`) using `sentence-transformers/all-MiniLM-L6-v2` and generates answers with a local Ollama LLM (`llama3.2:3b` by default).
- **Multi-Document FAISS Vector Store** — Upload and manage multiple PDFs simultaneously. Query across your entire knowledge base or scope queries to a single document via metadata filtering.
- **Smart PDF Text Normalization** — Cleans common PDF extraction artifacts (e.g., word-per-line breaks from Google Docs / Canva / macOS Quartz PDFs and Unicode ligatures) before chunking.
- **Reading-Order Context Assembly** — Retrieves the top-$k$ semantically similar chunks and sorts them in natural reading order (by document, page, and character offset) before prompting the LLM so timelines and multi-part sections remain coherent.
- **Transparent Citations & Diagnostics** — Every response includes exact source excerpts, page numbers, L2 distance scores, active compute device, and end-to-end latency.
- **Interactive Learning Playground** — Step-by-step Python scripts and a JupyterLab notebook (`backend/playground/`) walking through each stage of the RAG pipeline.

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Nuxt 3, Vue 3, Tailwind CSS (`:3000`) |
| **Backend API** | FastAPI, Uvicorn, Pydantic (`:8000`) |
| **Embeddings & ML** | PyTorch, Hugging Face `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Store** | FAISS (`faiss-cpu`) + LangChain |
| **LLM Engine** | Ollama (`llama3.2:3b` via `langchain-ollama`, `:11434`) |
| **Notebooks** | JupyterLab (`:8888`) |

---

## 📂 Project Structure

```text
SemanticDoc/
├── docker-compose.yml              # Orchestrates backend, frontend, and JupyterLab
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   └── main.py                 # FastAPI multi-PDF RAG engine
│   ├── playground/                 # Step-by-step RAG tutorials & Jupyter notebook
│   │   ├── 01_pytorch_device_check.py
│   │   ├── 02_hf_embeddings.py
│   │   ├── 03_create_sample_pdf.py
│   │   ├── 04_pdf_chunking.py
│   │   ├── 05_faiss_vector_store.py
│   │   ├── 06_rag_agent.py
│   │   └── SemanticDoc_Walkthrough.ipynb
│   ├── sample_data/                # Default sample PDF (handbook.pdf)
│   ├── uploads/                    # User-uploaded PDFs (persisted locally)
│   └── vector_store/               # Persisted FAISS index & document registry JSON
└── frontend/
    ├── Dockerfile
    ├── nuxt.config.ts
    ├── package.json
    └── app.vue                     # Nuxt 3 RAG UI (upload, doc filter, chat, citations)
```

---

## 🚀 Getting Started

### 1. Prerequisites

1. **Docker & Docker Compose** installed and running.
2. **Ollama** installed on your host machine ([ollama.com](https://ollama.com)).
3. Pull the default LLM model on your host machine and make sure Ollama is running:
   ```bash
   ollama pull llama3.2:3b
   ollama serve
   ```

### 2. Run with Docker Compose (Recommended)

From the project root:

```bash
docker compose up --build
```

Once started, the following services are available:
- **Frontend UI (Nuxt 3):** http://localhost:3000
- **Backend API (FastAPI):** http://localhost:8000 (Swagger docs at http://localhost:8000/docs)
- **JupyterLab Playground:** http://localhost:8888

---

## 💻 Local Development (Without Docker)

### Backend (FastAPI)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Point to local Ollama instance
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_MODEL="llama3.2:3b"

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend (Nuxt 3)

```bash
cd frontend
npm install
npm run dev
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Returns PyTorch version, compute device, active models, and FAISS index stats |
| `GET` | `/api/documents` | Lists all indexed PDF documents with page and chunk counts |
| `POST` | `/api/documents/upload` | Uploads a `.pdf` file (`multipart/form-data`), normalizes text, chunks, and indexes into FAISS |
| `DELETE` | `/api/documents/{filename}` | Removes a PDF's vectors from FAISS and deletes the uploaded file |
| `POST` | `/api/query` | Runs semantic search + LLM generation across all PDFs or filtered by `doc_name` |

### Example Query Request

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the remote work policy?",
    "top_k": 5,
    "doc_name": null
  }'
```

---

## 🧪 Playground & Walkthrough

The `backend/playground/` folder contains standalone scripts that break down how SemanticDoc works under the hood:

1. **`01_pytorch_device_check.py`** — Detects CUDA, Apple Silicon (`mps`), or CPU tensor acceleration.
2. **`02_hf_embeddings.py`** — Generates 384-dimensional normalized vector embeddings with Hugging Face.
3. **`03_create_sample_pdf.py`** — Generates a sample company handbook PDF using ReportLab.
4. **`04_pdf_chunking.py`** — Loads and splits PDFs with `RecursiveCharacterTextSplitter`.
5. **`05_faiss_vector_store.py`** — Builds, saves, and queries a local FAISS index.
6. **`06_rag_agent.py`** — Connects FAISS retrieval with local Ollama generation.
7. **`SemanticDoc_Walkthrough.ipynb`** — Interactive Jupyter notebook combining all steps.
