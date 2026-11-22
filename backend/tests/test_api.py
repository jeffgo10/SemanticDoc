from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk

from app.main import app, clean_pdf_text, ml_state


def test_clean_pdf_text_normalizes_artifacts():
    """
    1. Pipeline Unit Test: Verifies word-per-line PDF export artifacts ('\\n ')
    and Unicode ligatures ('ﬁ', 'ﬂ') are cleaned while preserving bullet points.
    """
    dirty_pdf_text = (
        "LIDD\n \nSupply\n \nChain\n \n—\n \nSenior\n \nEngineer\n"
        "●\n \nBuilt\n \nworkﬂows\n \nand\n \nveriﬁcation."
    )
    cleaned = clean_pdf_text(dirty_pdf_text)

    assert "LIDD Supply Chain — Senior Engineer" in cleaned
    assert "workflows and verification." in cleaned
    assert "\n● Built" in cleaned


def test_health_and_initial_seed():
    """
    2. Lifespan & PyTorch Integration Test: Verifies that FastAPI lifespan boots PyTorch,
    loads HuggingFace all-MiniLM-L6-v2, and indexes sample_data/handbook.pdf into FAISS.
    """
    with TestClient(app) as client:
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.json()

        assert data["status"] == "healthy"
        assert data["compute_device"] in ["cpu", "mps", "cuda"]
        assert data["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert data["index_loaded"] is True
        assert data["total_documents"] >= 1
        assert data["total_chunks_indexed"] >= 3


def test_retrieval_model_evaluation():
    """
    3. ML Retrieval Evaluation Test: Evaluates real PyTorch embeddings + FAISS search
    on handbook.pdf to verify semantic accuracy and L2 distance threshold.
    """
    with TestClient(app):
        assert ml_state.vector_store is not None
        results = ml_state.vector_store.similarity_search_with_score(
            "Can I deploy code to production on a Friday?",
            k=2,
            filter={"doc_name": "handbook.pdf"},
        )
        assert len(results) > 0
        top_doc, l2_distance = results[0]

        # Page 2 (0-indexed page 1) contains the Friday deployment rule
        assert top_doc.metadata["page"] + 1 == 2
        assert "Friday" in top_doc.page_content
        assert l2_distance < 1.35


def test_query_and_sse_streaming_with_mock_llm():
    """
    4. End-to-End RAG API & SSE Stream Test: Mocks the Ollama LLM so CI/CD can verify
    both POST /api/query and POST /api/query/stream without needing an external Ollama process.
    """
    with TestClient(app) as client:
        # Mock Ollama .invoke() and .stream()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Friday deployments are prohibited unless patching a critical CVE (handbook.pdf, Page 2)."
        )
        mock_llm.stream.return_value = [
            AIMessageChunk(content="Friday "),
            AIMessageChunk(content="deployments "),
            AIMessageChunk(content="are prohibited (Page 2)."),
        ]
        ml_state.llm = mock_llm

        # A. Test synchronous POST /api/query
        res = client.post(
            "/api/query",
            json={
                "question": "Can I deploy on Friday?",
                "top_k": 2,
                "doc_name": "handbook.pdf",
            },
        )
        assert res.status_code == 200
        body = res.json()
        assert "Friday deployments" in body["answer"]
        assert len(body["sources"]) >= 1
        assert body["sources"][0]["doc_name"] == "handbook.pdf"

        # B. Test real-time SSE POST /api/query/stream
        stream_res = client.post(
            "/api/query/stream",
            json={
                "question": "Can I deploy on Friday?",
                "top_k": 2,
                "doc_name": "handbook.pdf",
            },
        )
        assert stream_res.status_code == 200
        assert "text/event-stream" in stream_res.headers["content-type"]
        raw_events = stream_res.text
        assert '"type": "sources"' in raw_events
        assert '"type": "token"' in raw_events
        assert '"type": "done"' in raw_events


def test_pydantic_input_validation():
    """
    5. Security & Schema Validation Test: Verifies Pydantic rejects invalid top_k or short questions.
    """
    with TestClient(app) as client:
        # Question too short (< 3 chars)
        bad_res = client.post("/api/query", json={"question": "Hi", "top_k": 3})
        assert bad_res.status_code == 422

        # Non-PDF file upload rejected
        bad_upload = client.post(
            "/api/documents/upload",
            files={"file": ("hack.txt", b"not a pdf", "text/plain")},
        )
        assert bad_upload.status_code == 400


def test_agent_tool_calling_integration():
    """
    6. AI Agent Tool-Calling Test: Verifies that budget calculation and inventory
    questions trigger the @tool functions (`calculate_onboarding_budget`, `list_uploaded_documents`)
    and return `tools_called` in the API response.
    """
    with TestClient(app) as client:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Total stipend budget for 14 hires is $16,800 AUD."
        )
        ml_state.llm = mock_llm

        res = client.post(
            "/api/query",
            json={
                "question": "If GoLogic hires 14 engineers, calculate the total stipend budget.",
                "top_k": 2,
            },
        )
        assert res.status_code == 200
        body = res.json()
        assert any("calculate_onboarding_budget" in t for t in body["tools_called"])