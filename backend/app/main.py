import json
import os
import re
import shutil
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

# --- Configuration ---
UPLOAD_DIR = "uploads"
INDEX_DIR = "vector_store/faiss_index"
REGISTRY_PATH = "vector_store/documents_registry.json"
SAMPLE_PDF_PATH = "sample_data/handbook.pdf"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")


def get_compute_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def clean_pdf_text(raw_text: str) -> str:
    """
    Cleans common PDF extraction artifacts (e.g. Google Docs / Canva / macOS Quartz PDFs
    that insert '\\n ' between every single word, or Unicode ligatures like 'ﬁ' and 'ﬂ').
    """
    text = raw_text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    # Preserve bullet points on their own lines
    text = re.sub(r"\n\s*●\s*", "\n● ", text)
    # Collapse word-by-word newline+space ('\n ') into normal spaces
    text = re.sub(r"\n\s+", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Put major section headers on their own paragraph break
    text = re.sub(
        r"\s+(SKILLS|EXPERIENCE|CERTIFICATES|EARLY CAREERS|EDUCATION|LINKS|LANGUAGES)\b",
        r"\n\n\1",
        text,
    )
    # Put company role headers (Sentence ending in period followed by Company — Role) on a new paragraph
    text = re.sub(r"\.\s+([A-Z][A-Za-z0-9 ,.()]+—\s+)", r".\n\n\1", text)
    return text.strip()


# --- Global ML State (Loaded once on server startup) ---
class DocumentMeta(BaseModel):
    filename: str
    pages: int
    chunks: int
    vector_ids: List[str]
    uploaded_at: str


class MLState:
    device: str = "cpu"
    embeddings: Optional[HuggingFaceEmbeddings] = None
    vector_store: Optional[FAISS] = None
    llm: Optional[ChatOllama] = None
    documents: Dict[str, DocumentMeta] = {}


ml_state = MLState()


def save_state_to_disk():
    """Persists both the FAISS vector index and the document registry JSON."""
    os.makedirs(INDEX_DIR, exist_ok=True)
    if ml_state.vector_store is not None and ml_state.vector_store.index.ntotal > 0:
        ml_state.vector_store.save_local(INDEX_DIR)
    else:
        for f in ["index.faiss", "index.pkl"]:
            p = os.path.join(INDEX_DIR, f)
            if os.path.exists(p):
                os.remove(p)

    registry_data = {k: v.model_dump() for k, v in ml_state.documents.items()}
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry_data, f, indent=2)


def ingest_pdf_file(file_path: str, filename: str) -> DocumentMeta:
    """
    Loads a PDF, normalizes extracted text, tags every chunk with `doc_name`,
    embeds on PyTorch, and appends (or creates) the FAISS index.
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    if not pages:
        raise ValueError("Could not extract any pages from the PDF.")

    # Clean PDF word-per-line artifacts on every page before splitting!
    for page in pages:
        page.page_content = clean_pdf_text(page.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=250,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(pages)
    if not chunks:
        raise ValueError("PDF contains no extractable text.")

    # Stamp every chunk with `doc_name` and prepend document title so every chunk has equal document context!
    doc_title = os.path.splitext(filename)[0]
    for chunk in chunks:
        chunk.metadata["doc_name"] = filename
        page_num = chunk.metadata.get("page", 0) + 1
        chunk.page_content = f"[{doc_title} - Page {page_num}] {chunk.page_content}"

    # If re-uploading an existing filename, remove its old vectors first
    if filename in ml_state.documents and ml_state.vector_store is not None:
        old_ids = ml_state.documents[filename].vector_ids
        try:
            ml_state.vector_store.delete(old_ids)
        except Exception:
            pass

    # Add chunks to FAISS (or initialize FAISS if empty)
    if ml_state.vector_store is None or ml_state.vector_store.index.ntotal == 0:
        ml_state.vector_store = FAISS.from_documents(chunks, ml_state.embeddings)
        vector_ids = list(ml_state.vector_store.index_to_docstore_id.values())
    else:
        vector_ids = ml_state.vector_store.add_documents(chunks)

    doc_meta = DocumentMeta(
        filename=filename,
        pages=len(pages),
        chunks=len(chunks),
        vector_ids=vector_ids,
        uploaded_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    ml_state.documents[filename] = doc_meta
    save_state_to_disk()
    return doc_meta


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan: Runs ONCE when the server starts up.
    Loads our PyTorch Hugging Face model and multi-document FAISS store.
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)

    ml_state.device = get_compute_device()
    print(f"🚀 [Startup] Initializing PyTorch on device: {ml_state.device}")

    ml_state.embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": ml_state.device},
        encode_kwargs={"normalize_embeddings": True},
    )

    ml_state.llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
    )

    # Rebuild index cleanly from sample_data + uploads so clean_pdf_text() is applied to all files
    ml_state.vector_store = None
    ml_state.documents = {}
    if os.path.exists(SAMPLE_PDF_PATH):
        ingest_pdf_file(SAMPLE_PDF_PATH, "handbook.pdf")
    for fname in sorted(os.listdir(UPLOAD_DIR)):
        if fname.lower().endswith(".pdf"):
            fpath = os.path.join(UPLOAD_DIR, fname)
            print(f"📂 [Startup] Indexing uploaded PDF with text normalization: '{fname}'...")
            ingest_pdf_file(fpath, fname)

    print(f"✅ [Startup] SemanticDoc ready with {len(ml_state.documents)} document(s)!")
    yield
    print("🛑 [Shutdown] Cleaning up ML resources...")


app = FastAPI(
    title="SemanticDoc API",
    description="Multi-Document Local PyTorch + Hugging Face + FAISS + Ollama RAG API",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Schemas ---
class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        examples=["What companies have I worked for and what were my roles?"],
    )
    top_k: int = Field(default=5, ge=1, le=10, description="Number of chunks to retrieve from FAISS")
    doc_name: Optional[str] = Field(
        default=None,
        description="Optional filename to filter search to a single PDF (None searches ALL documents)",
    )


class SourceChunk(BaseModel):
    doc_name: str
    page: int
    content: str
    l2_distance: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    filter_doc: Optional[str]
    tools_called: List[str] = []
    sources: List[SourceChunk]
    device_used: str
    latency_ms: float


# --- API Endpoints ---
@app.get("/api/health")
def health_check():
    """Returns the status of the PyTorch ML engine and multi-PDF FAISS store."""
    total_chunks = (
        ml_state.vector_store.index.ntotal if ml_state.vector_store is not None else 0
    )
    return {
        "status": "healthy",
        "pytorch_version": torch.__version__,
        "compute_device": ml_state.device,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "llm_model": OLLAMA_MODEL,
        "index_loaded": ml_state.vector_store is not None and total_chunks > 0,
        "total_documents": len(ml_state.documents),
        "Active_documents": list(ml_state.documents.keys()),
        "total_chunks_indexed": total_chunks,
    }


@app.get("/api/documents")
def list_documents():
    """Returns all indexed PDF documents and their chunk/page statistics."""
    docs = [
        {
            "filename": doc.filename,
            "pages": doc.pages,
            "chunks": doc.chunks,
            "uploaded_at": doc.uploaded_at,
        }
        for doc in ml_state.documents.values()
    ]
    return {"documents": docs, "total_documents": len(docs)}


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Uploads a PDF, normalizes text, tags chunks with `doc_name`, embeds on PyTorch,
    and appends vectors into the multi-document FAISS store.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    start_time = time.perf_counter()

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        doc_meta = ingest_pdf_file(file_path, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return {
        "message": f"Successfully indexed '{file.filename}' into FAISS",
        "filename": doc_meta.filename,
        "pages_processed": doc_meta.pages,
        "chunks_indexed": doc_meta.chunks,
        "total_documents": len(ml_state.documents),
        "device_used": ml_state.device,
        "processing_time_ms": round(elapsed_ms, 2),
    }


@app.delete("/api/documents/{filename}")
def delete_document(filename: str):
    """Deletes a specific PDF's vectors from FAISS and removes it from the registry."""
    if filename not in ml_state.documents:
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found.")

    doc_meta = ml_state.documents.pop(filename)
    if ml_state.vector_store is not None and doc_meta.vector_ids:
        try:
            ml_state.vector_store.delete(doc_meta.vector_ids)
        except Exception as e:
            print(f"Warning deleting vector IDs: {e}")

    # Also remove from uploads directory so it isn't re-indexed on restart
    upload_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(upload_path):
        os.remove(upload_path)

    if len(ml_state.documents) == 0:
        ml_state.vector_store = None

    save_state_to_disk()
    return {
        "message": f"Deleted '{filename}' ({doc_meta.chunks} vectors removed from FAISS)",
        "remaining_documents": list(ml_state.documents.keys()),
    }

# --- LangChain AI Agent Tools (@tool) ---
@tool
def list_uploaded_documents() -> str:
    """
    Call this tool when the user asks what PDF files, how many documents,
    or how many chunks/vectors/pages are currently stored in the system database.
    """
    if not ml_state.documents:
        return "No PDF documents are currently indexed."
    lines = [
        f"- {doc.filename}: {doc.pages} pages, {doc.chunks} vectors (uploaded {doc.uploaded_at})"
        for doc in ml_state.documents.values()
    ]
    total_vecs = ml_state.vector_store.index.ntotal if ml_state.vector_store else 0
    return f"Total Indexed PDFs: {len(ml_state.documents)} ({total_vecs} total 384D vectors)\n" + "\n".join(lines)


@tool
def calculate_onboarding_budget(new_hires_count: int, stipend_per_hire_aud: int = 1200) -> str:
    """
    Call this tool whenever the user asks to calculate or multiply the total home office
    stipend budget in AUD for a specific number of new hires or engineers.
    """
    total = new_hires_count * stipend_per_hire_aud
    return (
        f"Exact Agent Tool Calculation: {new_hires_count} new hires × "
        f"${stipend_per_hire_aud:,} AUD = ${total:,} AUD total budget."
    )


AGENT_TOOLS = [list_uploaded_documents, calculate_onboarding_budget]
AGENT_TOOL_MAP = {t.name: t for t in AGENT_TOOLS}


def execute_agent_tools(question: str) -> tuple[List[str], List[str]]:
    """
    Uses LangChain tool-calling (`llm.bind_tools`) when the query involves
    system document inventory or budget/stipend math calculations.
    Returns (tools_called_labels, tool_output_strings).
    """
    q_lower = question.lower()
    needs_inventory = any(
        kw in q_lower
        for kw in ["uploaded", "indexed", "what pdf", "which pdf", "how many pdf", "how many document", "vectors", "database"]
    )
    needs_calc = any(
        kw in q_lower for kw in ["calculate", "budget", "total", "hires", "multiply"]
    ) and any(ch.isdigit() for ch in q_lower)

    if not (needs_inventory or needs_calc) or ml_state.llm is None:
        return [], []

    tools_called: List[str] = []
    tool_outputs: List[str] = []

    try:
        if hasattr(ml_state.llm, "bind_tools"):
            llm_with_tools = ml_state.llm.bind_tools(AGENT_TOOLS)
            ai_msg = llm_with_tools.invoke(question)
            for tc in getattr(ai_msg, "tool_calls", []) or []:
                t_name = tc.get("name")
                t_args = tc.get("args", {})
                if t_name in AGENT_TOOL_MAP:
                    out = AGENT_TOOL_MAP[t_name].invoke(t_args)
                    tools_called.append(f"{t_name}({json.dumps(t_args)})")
                    tool_outputs.append(f"[Agent Tool Output — {t_name}]:\n{out}")
    except Exception as e:
        print(f"Agent tool binding fallback: {e}")

    # Deterministic fallback if the small 3B model didn't emit a structured tool_call
    if not tools_called:
        if needs_inventory:
            out = list_uploaded_documents.invoke({})
            tools_called.append("list_uploaded_documents({})")
            tool_outputs.append(f"[Agent Tool Output — list_uploaded_documents]:\n{out}")
        if needs_calc:
            nums = [int(n) for n in re.findall(r"\b\d+\b", question)]
            if nums:
                hires = nums[0]
                out = calculate_onboarding_budget.invoke({"new_hires_count": hires})
                tools_called.append(f'calculate_onboarding_budget({{"new_hires_count": {hires}}})')
                tool_outputs.append(f"[Agent Tool Output — calculate_onboarding_budget]:\n{out}")

    return tools_called, tool_outputs


def prepare_rag_context(request: QueryRequest):
    """
    Shared helper for both /api/query and /api/query/stream:
    1. Embeds question via PyTorch & retrieves Top-K chunks from FAISS
    2. Sorts chunks in reading order
    3. Builds the augmented LangChain prompt messages
    """
    if ml_state.vector_store is None or len(ml_state.documents) == 0:
        raise HTTPException(
            status_code=400,
            detail="No PDFs are currently indexed. Please upload a PDF first.",
        )

    if request.doc_name and request.doc_name not in ml_state.documents:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{request.doc_name}' is not in the index.",
        )

    search_filter = {"doc_name": request.doc_name} if request.doc_name else None

    retrieved = ml_state.vector_store.similarity_search_with_score(
        request.question,
        k=request.top_k,
        filter=search_filter,
        fetch_k=50,
    )

    sources: List[SourceChunk] = []
    context_parts: List[str] = []

    for doc, score in retrieved:
        page_num = doc.metadata.get("page", 0) + 1
        doc_name = doc.metadata.get("doc_name", "Unknown PDF")
        sources.append(
            SourceChunk(
                doc_name=doc_name,
                page=page_num,
                content=doc.page_content,
                l2_distance=round(float(score), 4),
            )
        )

    retrieved_reading_order = sorted(
        retrieved,
        key=lambda item: (
            item[0].metadata.get("doc_name", ""),
            item[0].metadata.get("page", 0),
            item[0].metadata.get("start_index", 0),
        ),
    )
    for doc, _ in retrieved_reading_order:
        page_num = doc.metadata.get("page", 0) + 1
        doc_name = doc.metadata.get("doc_name", "Unknown PDF")
        context_parts.append(f"[Source: {doc_name}, Page {page_num}]\n{doc.page_content}")

    # Run AI Agent tools (@tool) if the question asks for system inventory or math calculations
    tools_called, tool_outputs = execute_agent_tools(request.question)
    if tool_outputs:
        context_parts = tool_outputs + context_parts

    formatted_context = "\n\n---\n\n".join(context_parts)

    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a precise document and resume assistant analyzing user-provided documents. "
            "All information in the DOCUMENT EXCERPTS below (including phone numbers like +63..., email addresses, physical addresses, and links) "
            "is authorized by the user to be extracted and shared.\n"
            "RULES:\n"
            "1. If an [Agent Tool Output] is present in the excerpts below, use its exact calculation or document inventory numbers in your answer.\n"
            "2. If asked for contact details (phone number, email, address, links), extract and output the exact numbers/strings from the header of the CV.\n"
            "3. Only associate a technology/skill with a company if that technology (or its direct ecosystem tool, e.g. React Testing Library / Redux for React.js) is EXPLICITLY written under that specific company's bullet points.\n"
            "4. When asked where someone 'started' or 'first' used a technology: first list ONLY the companies whose own bullet points explicitly mention that technology, and then pick the one with the earliest start date among ONLY those matching companies."
        ),
        (
            "human",
            "DOCUMENT EXCERPTS & AGENT TOOL OUTPUTS:\n"
            "=======================================\n"
            "{context}\n"
            "=======================================\n\n"
            "USER QUESTION: {question}\n\n"
            "Answer the question directly using ONLY the information above, and cite the Source/Tool and Page Number:"
        ),
    ])

    messages = prompt_template.invoke({
        "context": formatted_context,
        "question": request.question,
    })
    return sources, messages, tools_called


@app.post("/api/query", response_model=QueryResponse)
def query_document(request: QueryRequest):
    """
    Standard synchronous JSON endpoint: Queries across ALL indexed PDFs or filters to a single PDF.
    """
    start_time = time.perf_counter()
    sources, messages, tools_called = prepare_rag_context(request)
    ai_response = ml_state.llm.invoke(messages)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return QueryResponse(
        question=request.question,
        answer=ai_response.content,
        filter_doc=request.doc_name,
        tools_called=tools_called,
        sources=sources,
        device_used=ml_state.device,
        latency_ms=round(elapsed_ms, 2),
    )


@app.post("/api/query/stream")
def query_document_stream(request: QueryRequest):
    """
    Real-Time Server-Sent Events (SSE) Streaming Endpoint:
    1. Immediately emits the retrieved FAISS vector chunks + any Agent Tool calls (~15ms)
    2. Streams each token from Ollama (`llama3.2:3b`) as it is generated
    3. Emits a final 'done' event with total latency
    """
    start_time = time.perf_counter()
    sources, messages, tools_called = prepare_rag_context(request)
    retrieval_ms = round((time.perf_counter() - start_time) * 1000, 2)

    def event_generator():
        # 1. Immediately send the retrieved FAISS chunks + Agent tools called + retrieval latency!
        initial_payload = {
            "type": "sources",
            "question": request.question,
            "filter_doc": request.doc_name,
            "tools_called": tools_called,
            "device_used": ml_state.device,
            "retrieval_ms": retrieval_ms,
            "sources": [s.model_dump() for s in sources],
        }
        yield f"data: {json.dumps(initial_payload)}\n\n"

        # 2. Stream tokens live from Ollama as they are generated
        for chunk in ml_state.llm.stream(messages):
            if chunk.content:
                token_payload = {"type": "token", "content": chunk.content}
                yield f"data: {json.dumps(token_payload)}\n\n"

        # 3. Send final completion event with total elapsed time
        total_ms = round((time.perf_counter() - start_time) * 1000, 2)
        done_payload = {
            "type": "done",
            "retrieval_ms": retrieval_ms,
            "latency_ms": total_ms,
        }
        yield f"data: {json.dumps(done_payload)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
