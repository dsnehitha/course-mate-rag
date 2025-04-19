import os
import torch
import hashlib
import pickle
import time
from ollama import Client
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant
from qdrant_client.http import models

#using all 3 open source models - embedding, llm and vector store

DATA_DIRECTORY = "./course_materials/"
METADATA_DIR = "./course_material_vectore_store"
METADATA_FILE = os.path.join(METADATA_DIR, "metadata.pkl")
COLLECTION_NAME = "student_coursework"
QDRANT_URL = "http://localhost:6333"

os.makedirs(METADATA_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

index = None
chunks = None

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": device})

# Initialize Ollama client
ollama_client = Client(
  host='http://localhost:11434',
  headers={'x-some-header': 'some-value'}
)

# Initialize Qdrant client
q_client = QdrantClient(url=QDRANT_URL)

# Create a collection in Qdrant
vectore_store = Qdrant(
    client = q_client,
    collection_name=COLLECTION_NAME,
    embeddings=embedding_model,
)

# Function to compute a hash (fingerprint) of all PDFs
def compute_pdf_hash():
    hasher = hashlib.sha256()
    pdf_files = sorted([f for f in os.listdir(DATA_DIRECTORY) if f.endswith(".pdf")])

    for pdf in pdf_files:
        with open(os.path.join(DATA_DIRECTORY, pdf), "rb") as f:
            hasher.update(f.read())

    return hasher.hexdigest()

# Function to check if Qdrant needs rebuilding
def is_db_outdated():
    if not os.path.exists(METADATA_FILE):
        return True
    try:
        with open(METADATA_FILE, "rb") as f:
            saved_hash = pickle.load(f).get("pdf_hash", None)

        current_hash = compute_pdf_hash()

        return saved_hash != current_hash
    
    except Exception as e:
        return True
    
def build_collection():
    global index, chunks
    
    start_time = time.time()

    if not is_db_outdated():
        print("No changes detected in PDFs.")
        return
    
    print("Building collection of documents...")

    """Builds a new collection of documents."""
    # Load documents from the PDF directory
    loader = DirectoryLoader(DATA_DIRECTORY, glob="**/*.pdf", loader_cls=PyPDFLoader, use_multithreading=True)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    print(f"{len(chunks)} text chunks extracted.")

    # Generate embeddings for each chunk
    # chunk_texts = [chunk.page_content for chunk in chunks]
    # chunk_embeddings = embeddings.embed_documents(chunk_texts)
    existing_collections = [col.name for col in q_client.get_collections().collections]
    
    if "student_coursework" not in existing_collections:
        print("Creating collection: student_coursework")
        q_client.create_collection(
            collection_name="student_coursework",
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)  # Adjust size based on embedding model
        )
    else:
        print("Updating collection: student_coursework")
        q_client.update_collection(
            collection_name="student_coursework",
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=10000),
        )

    vectore_store.add_documents(chunks)

    # Save metadata including the hash of the PDFs
    with open(METADATA_FILE, "wb") as f:
        pickle.dump({"pdf_hash": compute_pdf_hash()}, f)
    
    end_time = time.time()
    print(f"Collection built in {end_time - start_time:.2f} seconds.")

def generate_answer(query):
    """Generates an answer to a query using the Qdrant collection."""
    # Retrieve relevant documents from the collection
    results = vectore_store.similarity_search(query, k=5)

    # Concatenate the retrieved documents' content
    context = "\n".join([doc.page_content for doc in results])

    print("\ncontext\n",context)

    prompt = f"Use the provided course material to answer:\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    # Generate an answer using the Ollama model
    response = ollama_client.chat(
        model="llama3.2",
        messages=[
        {"role": "system", "content": "You are a helpful course assistant."},
        {"role": "user", "content": prompt}])

    return response['message']['content']

def main():
    build_collection()
    
    query = input("Enter your question: ").strip()

    start_time = time.time()
     
    answer = generate_answer(query)
    print(f"\nAnswer:\n{answer}")

    end_time = time.time()

    print(f"\nQuestion answered in {end_time - start_time:.2f} seconds")
    if torch.cuda.is_available():
        print(f"Answer generated using {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB of GPU memory.")

if __name__ == "__main__":
    main()
