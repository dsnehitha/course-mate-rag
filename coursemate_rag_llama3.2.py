import os
import time
import faiss
import pickle
import hashlib
import numpy as np
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from ollama import Client
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

device = "cuda" if torch.cuda.is_available() else "cpu"

client = Client(
  host='http://localhost:11434',
  headers={'x-some-header': 'some-value'}
)

# Path to store FAISS index
PDF_PATH = "./course_materials/"
DB_PATH = "./course_material_faiss"
METADATA_FILE = f"{DB_PATH}/metadata.pkl"

index = None
chunks = None
embedding_model = OpenAIEmbeddings(openai_api_key="")

# Function to compute a hash (fingerprint) of all PDFs
def compute_pdf_hash():
    hasher = hashlib.sha256()
    pdf_files = sorted([f for f in os.listdir(PDF_PATH) if f.endswith(".pdf")])

    for pdf in pdf_files:
        with open(os.path.join(PDF_PATH, pdf), "rb") as f:
            hasher.update(f.read())

    return hasher.hexdigest()

# Function to check if FAISS needs rebuilding
def is_faiss_outdated():
    if not os.path.exists(METADATA_FILE):
        return True
    try:
        with open(METADATA_FILE, "rb") as f:
            saved_hash = pickle.load(f).get("pdf_hash", None)

        current_hash = compute_pdf_hash()
        return saved_hash != current_hash
    
    except Exception as e:
        return True
    
def process_document(doc):
    """Splits a single document into text chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    return text_splitter.split_documents([doc])

# **Parallelized Function to Generate Embeddings**
def generate_embedding(text):
    """Generates embeddings for a given text chunk."""
    embedding = np.array(embedding_model.embed_query(text), dtype=np.float32)
    # return torch.tensor(embedding, device=device)
    return embedding

# Load Course Materials & Create FAISS Index
def build_faiss_index():
    global index, chunks

    start_time = time.time()

    if not is_faiss_outdated():
        print("FAISS is up-to-date. No rebuild needed.")
        return

    print("Building FAISS index...")

    loader = DirectoryLoader(PDF_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    # Split Text into Chunks
    with ThreadPoolExecutor() as executor:
        chunk_lists = list(executor.map(process_document, documents))

    chunks = [chunk for sublist in chunk_lists for chunk in sublist]

    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    # chunks = text_splitter.split_documents(documents)
    
    texts = [chunk.page_content for chunk in chunks]

    print(f"{len(chunks)} text chunks extracted.")

    with ProcessPoolExecutor() as executor:
        vectors = list(executor.map(generate_embedding, texts))
    # vectors = [generate_embedding(text) for text in texts]
    vectors = np.array(vectors).astype("float32")

    # # Generate Embeddings
    # vectors = [embedding_model.embed_query(chunk.page_content) for chunk in chunks]

    # vectors = torch.stack(vectors).cpu().numpy()

    print(f"Embeddings generated for {len(vectors)} chunks.")

    # Create FAISS Index
    d = vectors.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(vectors)

    print(f"FAISS Index trained with {len(vectors)} clusters.")

    # Save Index
    os.makedirs(DB_PATH, exist_ok=True)
    faiss.write_index(index, f"{DB_PATH}/faiss.index")
    with open(f"{DB_PATH}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    with open(METADATA_FILE, "wb") as f:
        pickle.dump({"pdf_hash": compute_pdf_hash()}, f)

    print("FAISS Index Built!")

    end_time = time.time()
    print(f"FAISS Index Built in {end_time - start_time:.2f} seconds (GPU Accelerated)")

# Load FAISS Index
def load_faiss_index():
    global index, chunks

    start_time = time.time()

    if index is None or chunks is None:
        # Loading FAISS index into memory
        index = faiss.read_index(f"{DB_PATH}/faiss.index")
        with open(f"{DB_PATH}/chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
    
    end_time = time.time()
    print(f"FAISS Index Loaded in {end_time - start_time:.2f} seconds")

# Retrieve Relevant Chunks
def retrieve_top_chunks(query, k=3):
    load_faiss_index()

    query_vector = embedding_model.embed_query(query)

    query_vector = np.array([query_vector], dtype=np.float32)

    _, indices = index.search(query_vector, k) #neighbors of the query vector

    return " ".join([chunks[i].page_content for i in indices[0] if i < len(chunks)])

# Generate Answer with llama3.2
def generate_answer(query):
    context = retrieve_top_chunks(query)
    print("\ncontext\n", context)
    prompt = f"Use the provided course material to answer:\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    response = client.chat(model='llama3.2',  messages=[
        {"role": "system", "content": "You are a helpful course assistant."},
        {"role": "user", "content": prompt}])
    
    return response.message.content

# Main function to handle command-line queries
def main():
    # Build FAISS Index
    build_faiss_index()
    
    query = input("Enter your question: ").strip()

    start_time = time.time()
    # Get Answer
    answer = generate_answer(query)
    print(f"\nAnswer:\n{answer}")

    end_time = time.time()
    print(f"Question answered in {end_time - start_time:.2f} seconds (GPU Accelerated)")


if __name__ == "__main__":
    main()