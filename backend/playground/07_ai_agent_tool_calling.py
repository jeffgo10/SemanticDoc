import json
import os
import torch
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

INDEX_DIR = "vector_store/faiss_index"
FALLBACK_INDEX_DIR = "vector_store/handbook_index"
REGISTRY_PATH = "vector_store/documents_registry.json"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")


def get_compute_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Initialize PyTorch Embeddings & FAISS once
device = get_compute_device()
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)
active_index_dir = INDEX_DIR if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")) else FALLBACK_INDEX_DIR
vector_store = FAISS.load_local(
    active_index_dir,
    embeddings,
    allow_dangerous_deserialization=True,
)


# =====================================================================
# 1. DEFINE PYTHON TOOLS (@tool) THAT THE LLM AGENT CAN CHOOSE TO CALL
#    Note: The docstring is CRITICAL! The LLM reads the docstring to
#    understand when and how to call each tool!
# =====================================================================

@tool
def list_uploaded_documents() -> str:
    """
    Call this tool when the user asks what PDF files, how many documents,
    or how many chunks/pages are currently stored in the system database.
    """
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r") as f:
            registry = json.load(f)
        summary = [
            f"- {name}: {meta['pages']} pages, {meta['chunks']} vectors"
            for name, meta in registry.items()
        ]
        return "Indexed PDF Documents:\n" + "\n".join(summary)
    return f"FAISS index loaded with {vector_store.index.ntotal} vectors."


@tool
def search_faiss_knowledge_base(search_query: str) -> str:
    """
    Call this tool to search inside the PDF documents (such as the GoLogic Engineering
    Handbook or Jeffrey Go's CV) for company policies, stipends, deployments, or work experience.
    """
    results = vector_store.similarity_search_with_score(search_query, k=3)
    excerpts = []
    for doc, score in results:
        doc_name = doc.metadata.get("doc_name", "handbook.pdf")
        page = doc.metadata.get("page", 0) + 1
        excerpts.append(f"[{doc_name} - Page {page} | L2 dist: {score:.3f}]: {doc.page_content}")
    return "\n\n".join(excerpts)


@tool
def calculate_onboarding_budget(new_hires_count: int, stipend_per_hire_aud: int = 1200) -> str:
    """
    Call this tool whenever the user asks to calculate or multiply the total home office
    stipend budget in AUD for a specific number of new hires.
    """
    total = new_hires_count * stipend_per_hire_aud
    return f"Exact calculation: {new_hires_count} new hires × ${stipend_per_hire_aud:,} AUD = ${total:,} AUD"


# Map tool names to actual Python functions for execution
TOOLS = [list_uploaded_documents, search_faiss_knowledge_base, calculate_onboarding_budget]
TOOL_MAP = {t.name: t for t in TOOLS}


# =====================================================================
# 2. THE AI AGENT LOOP (LLM Decides Tool -> Python Runs Tool -> LLM Answers)
# =====================================================================

def run_agent(user_question: str):
    print("\n" + "=" * 80)
    print(f"🙋 USER QUESTION: {user_question}")
    print("-" * 80)

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
    )
    # Bind our 3 Python @tool functions to Llama 3.2!
    llm_with_tools = llm.bind_tools(TOOLS)

    messages = [
        SystemMessage(
            content=(
                "You are an autonomous AI Agent for SemanticDoc. "
                "Always use the provided tools to look up facts, check uploaded PDFs, or calculate numbers "
                "before answering the user's question."
            )
        ),
        HumanMessage(content=user_question),
    ]

    # STEP A: Ask the LLM what to do first!
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)

    # Check if the LLM decided to call one or more Python tools!
    if ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"🛠️  [AGENT DECISION] LLM chose to call tool: `{tool_name}({tool_args})`")

            # STEP B: Python executes the tool the LLM asked for
            selected_tool = TOOL_MAP[tool_name]
            tool_output = selected_tool.invoke(tool_args)
            print(f"📦 [TOOL OUTPUT]   Returned to LLM:\n   {tool_output[:200]}...")

            # Append the tool's output as a ToolMessage so the LLM can read it
            messages.append(
                ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
            )

        # STEP C: Now that the LLM has the tool output, it generates the final answer!
        final_response = llm_with_tools.invoke(messages)
        print(f"\n🤖 [FINAL AGENT ANSWER]:\n{final_response.content}")
    else:
        print("ℹ️  (LLM answered directly without calling a tool)")
        print(f"\n🤖 [FINAL AGENT ANSWER]:\n{ai_msg.content}")


if __name__ == "__main__":
    # Test 1: Should autonomously pick `list_uploaded_documents`
    run_agent("What PDF files are currently uploaded in our system database and how many vectors do they have?")

    # Test 2: Should autonomously pick `search_faiss_knowledge_base` (RAG as a Tool!)
    run_agent("According to the engineering handbook, what is the rule for deploying on a Friday?")

    # Test 3: Should autonomously pick `calculate_onboarding_budget`
    run_agent("If GoLogic hires 14 new engineers next month, what is the total home office stipend budget in AUD?")