from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main():
    pdf_path = "sample_data/handbook.pdf"
    print(f"1️⃣  Loading PDF: {pdf_path}...")

    # 1. Load the PDF (PyPDFLoader creates one Document object per page)
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"    Loaded {len(pages)} pages from PDF.")

    # Inspect what a raw LangChain 'Document' looks like
    print(f"    Sample Page 1 Metadata: {pages[0].metadata}")

    # 2. Split Pages into Overlapping Chunks
    # RecursiveCharacterTextSplitter tries to split cleanly on paragraphs ("\n\n"),
    # then lines ("\n"), then spaces (" ") so it doesn't slice words in half.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,       # Max characters per chunk
        chunk_overlap=50,     # Overlapping characters between consecutive chunks
        add_start_index=True  # Tracks exact character offset where chunk starts in the page!
    )

    chunks = splitter.split_documents(pages)
    print(f"\n2️⃣  Split {len(pages)} pages into {len(chunks)} smaller chunks!\n" + "=" * 70)

    # 3. Inspect every chunk and its metadata
    for i, chunk in enumerate(chunks):
        page_num = chunk.metadata.get("page", 0) + 1  # PyPDF pages are 0-indexed
        start_idx = chunk.metadata.get("start_index", 0)
        print(f"📦 CHUNK #{i} | Page {page_num} (char offset {start_idx}) | Length: {len(chunk.page_content)} chars")
        print(f"   Text: {repr(chunk.page_content)}")
        print("-" * 70)


if __name__ == "__main__":
    main()