import torch
from sentence_transformers import SentenceTransformer, util


def get_compute_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    device = get_compute_device()
    print(f"1️⃣  Initializing PyTorch on device: '{device}'")

    # 2. Download & Load the Hugging Face Model into PyTorch
    # First run downloads ~90MB from Hugging Face Hub into /app/.cache/huggingface
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"2️⃣  Loading Hugging Face model: {model_name}...")
    model = SentenceTransformer(model_name, device=device)

    # Prove that this Hugging Face model IS a PyTorch Neural Network (torch.nn.Module)!
    print(f"    Model Python Type : {type(model)}")
    print(f"    Is PyTorch Module?: {isinstance(model, torch.nn.Module)}")

    # 3. Imagine these 3 chunks came from a company policy PDF:
    pdf_chunks = [
        "Employees may request up to 15 days of paid vacation per calendar year.",
        "All production database passwords must be rotated every 90 days using Vault.",
        "The office espresso machine is cleaned every Friday afternoon at 4 PM.",
    ]

    # And this is the user's question (notice it doesn't use the words 'vacation' or 'paid'!):
    # user_question = "How much time off do I get?"
    # user_question = "How often do we need to change DB credentials?"
    user_question = "where do i get coffee??"

    # 4. Convert text into PyTorch Tensors (Embeddings)
    # convert_to_tensor=True tells Hugging Face to return raw PyTorch Tensors (torch.Tensor)
    print("\n3️⃣  Encoding text into PyTorch Tensors...")
    chunk_tensors = model.encode(pdf_chunks, convert_to_tensor=True)
    question_tensor = model.encode(user_question, convert_to_tensor=True)

    # Inspect the raw PyTorch Tensor!
    print(f"    question_tensor type  : {type(question_tensor)}")
    print(f"    question_tensor shape : {question_tensor.shape}  <-- 384 dimensions!")
    print(f"    chunk_tensors shape   : {chunk_tensors.shape}  <-- 3 chunks x 384 dimensions!")
    print(f"    First 5 numbers of question_tensor: {question_tensor[:5].tolist()}")

    # 5. Calculate Cosine Similarity using PyTorch!
    # Score ranges from -1.0 (opposite) to 1.0 (identical meaning)
    cosine_scores = util.cos_sim(question_tensor, chunk_tensors)[0]

    print(f"\n4️⃣  Semantic Search Results for: '{user_question}'\n" + "-" * 60)
    for i, chunk in enumerate(pdf_chunks):
        score = cosine_scores[i].item()  # .item() extracts a Python float from a 1-element PyTorch Tensor
        print(f"Score: {score:.4f} | Chunk: {chunk}")

    # Find the winning chunk using PyTorch's argmax()
    best_idx = torch.argmax(cosine_scores).item()
    print("-" * 60)
    print(f"🏆 Best Match (Index {best_idx}): '{pdf_chunks[best_idx]}'")


if __name__ == "__main__":
    main()
