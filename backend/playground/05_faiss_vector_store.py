import os
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

INDEX_DIR = "vector_store/handbook_index"


def get_compute_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_embedding_model() -> HuggingFaceEmbeddings:
    device = get_compute_device()
    print(f"🧠 Loading Hugging Face Embeddings on PyTorch device: '{device}'...")

    # Wrap sentence-transformers/all-MiniLM-L6-v2 in LangChain's HuggingFaceEmbeddings
    # normalize_embeddings=True makes all vectors length 1.0 so L2 distance == Cosine Similarity!
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(pdf_path: str, embeddings: HuggingFaceEmbeddings) -> FAISS:
    print(f"\n1️⃣  Ingesting & chunking PDF: {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        add_start_index=True,
    )
    chunks = splitter.split_documents(pages)
    print(f"    Created {len(chunks)} chunks from {len(pages)} pages.")

    print("2️⃣  Embedding chunks with PyTorch & indexing into FAISS...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Save the FAISS index to disk so we don't have to re-embed next time!
    os.makedirs(INDEX_DIR, exist_ok=True)
    vector_store.save_local(INDEX_DIR)
    print(f"✅ Saved FAISS index to disk at: '{INDEX_DIR}/'")
    return vector_store


def search_vector_store(embeddings: HuggingFaceEmbeddings):
    print(f"\n3️⃣  Loading saved FAISS index from '{INDEX_DIR}/'...")
    # allow_dangerous_deserialization=True is required by LangChain to load the local .pkl metadata file we just created
    vector_store = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    # Let's test 3 questions that use different wording from the PDF!
    test_questions = [
        "Can I deploy code to production on a Friday?",
        "How much money do I get to buy a monitor for home?",
        "How fast do I need to respond to a PagerDuty alert?",
    ]

    for question in test_questions:
        print("\n" + "=" * 75)
        print(f"🔎 QUESTION: '{question}'")
        print("-" * 75)

        # Retrieve the Top 2 most relevant chunks from FAISS
        results = vector_store.similarity_search_with_score(question, k=2)

        for rank, (doc, distance_score) in enumerate(results, start=1):
            page_num = doc.metadata.get("page", 0) + 1
            # Note: FAISS returns L2 distance (Lower = Closer match! 0.0 is identical)
            print(f"  Rank #{rank} | L2 Distance: {distance_score:.4f} | Source: Page {page_num}")
            print(f"  Content : {doc.page_content}")
            print()


if __name__ == "__main__":
    embedding_model = get_embedding_model()
    build_vector_store("sample_data/handbook.pdf", embedding_model)
    search_vector_store(embedding_model)