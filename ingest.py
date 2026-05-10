import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# --- STEP 1: Read the PDF ---
def load_pdf(pdf_path):
    print(f"Reading PDF: {pdf_path}")
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    print(f"✅ Extracted {len(reader.pages)} pages")
    return text

# --- STEP 2: Split text into chunks ---
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)
    print(f"✅ Split into {len(chunks)} chunks")
    return chunks

# --- STEP 3: Store chunks in ChromaDB ---
def store_in_chromadb(chunks):
    print("Generating embeddings and storing in ChromaDB...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print(f"✅ Stored {len(chunks)} chunks in ChromaDB!")
    return vectordb

# --- MAIN ---
if __name__ == "__main__":
    pdf_folder = "./pdfs"
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            path = os.path.join(pdf_folder, filename)
            text = load_pdf(path)
            chunks = split_text(text)
            store_in_chromadb(chunks)
    print("\n🎉 Ingestion complete! Your PDF is now stored in ChromaDB.")
