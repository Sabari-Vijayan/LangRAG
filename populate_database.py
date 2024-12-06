from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings  # Replace OpenAI embeddings
from langchain_community.vectorstores import Chroma
import os
import shutil

# Remove OpenAI-specific imports since we're not using them anymore

CHROMA_PATH = "chroma"
DATA_PATH = "data"  # Directory containing PDF files

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    # Use PyPDFLoader to load PDF files
    pdf_files = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    documents = []
    
    for pdf_path in pdf_files:
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    
    print(f"Loaded {len(documents)} pages from PDF files.")
    return documents

def split_text(documents: list[Document]):
    # Text splitting remains the same
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Optional: Print a sample chunk to verify content
    if chunks:
        print("Sample chunk:")
        print(chunks[0].page_content)
        print(chunks[0].metadata)

    return chunks

def save_to_chroma(chunks: list[Document]):
    # Clear out the existing database
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create embeddings using a free, local model
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",  # Lightweight, efficient embedding model
        model_kwargs={'device': 'cpu'},  # Use CPU (change to 'cuda' if you have GPU)
        encode_kwargs={'normalize_embeddings': True}  # Normalize vector embeddings
    )

    # Create and persist the Chroma vector store
    db = Chroma.from_documents(
        chunks, 
        embeddings,  # Use local Hugging Face embeddings
        persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()