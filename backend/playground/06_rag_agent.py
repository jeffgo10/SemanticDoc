import os
import torch
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

INDEX_DIR = "vector_store/handbook_index"
# Inside Docker on Mac, 'host.docker.internal' points to your Mac host!
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")


def get_compute_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    # 1. Load our PyTorch + Hugging Face Embedding Model & Saved FAISS Database
    device = get_compute_device()
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_store = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    # 2. Initialize the Local LLM (Ollama)
    # temperature=0.0 makes the LLM strictly factual and deterministic (no creative guessing!)
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
    )

    # 3. The "A" in RAG: Augmented Prompt Engineering
    # We strictly instruct the LLM to ONLY use the retrieved FAISS chunks!
    # 3. The "A" in RAG: Augmented Prompt Engineering
    # Putting {context} and {question} together in the human message works much better for 1B-3B models!
    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant that answers questions based strictly on the provided handbook excerpts. "
            "Always mention the Page Number in your answer."
        ),
        (
            "human",
            "Read the following excerpts from the GoLogic Engineering Handbook:\n\n"
            "{context}\n\n"
            "Question: {question}\n"
            "Instruction: Answer the question using the excerpts above. "
            "Only if the excerpts are completely unrelated to the question, reply: "
            "'I do not have enough information in the document to answer that.'"
        ),
    ])

    # 4. Test 3 Questions (2 answerable across pages, plus 1 trick question NOT in the PDF!)
    questions = [
        "If there is a critical security bug on a Friday, can I deploy to production?",
        "How much is the home office stipend and what can I buy with it?",
        "What is the company policy on bringing dogs to the office?",  # Trick question! Not in PDF!
    ]

    for question in questions:
        print("\n" + "=" * 80)
        print(f"🙋 USER QUESTION: {question}")

        # --- STEP A: RETRIEVAL (PyTorch + FAISS) ---
        retrieved_docs = vector_store.similarity_search_with_score(question, k=3)

        # Format the retrieved chunks with their Page Numbers so the LLM can read & cite them
        context_parts = []
        sources = []
        for doc, score in retrieved_docs:
            page_num = doc.metadata.get("page", 0) + 1
            sources.append(f"Page {page_num} (L2 dist: {score:.3f})")
            context_parts.append(f"[Excerpt from Page {page_num}]: {doc.page_content}")

        formatted_context = "\n\n".join(context_parts)
        print(f"📚 RETRIEVED SOURCES: {', '.join(sources)}")
        print(f"📄 CONTEXT SENT TO LLM:\n{formatted_context}")

        # --- STEP B & C: AUGMENTATION & GENERATION (LLM) ---
        messages = prompt_template.invoke({
            "context": formatted_context,
            "question": question,
        })
        response = llm.invoke(messages)

        print(f"\n🤖 AI ANSWER:\n{response.content}")


if __name__ == "__main__":
    main()
    