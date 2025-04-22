import os
import torch
import hashlib
import pickle
import time
from ollama import chat
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant
from qdrant_client.http import models

DATA_DIR = "./course_materials"
METADATA_DIR = "./course_materials/metadata"
METADATA_FILE = os.path.join(METADATA_DIR, "metadata.pkl")
COLLECTION_NAME = "nasa_course_materials"
QDRANT_URL = "http://localhost:6333"

os.makedirs(METADATA_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

index = None
chunks = None

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": device})

q_client = QdrantClient(url=QDRANT_URL, prefer_grpc=False)

vector_store = Qdrant(
    client=q_client,
    collection_name=COLLECTION_NAME,
    embeddings=embedding_model
)


#helper functions
def compute_pdf_hash():
    hasher = hashlib.sha256()
    pdf_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')])

    for pdf in pdf_files:
        with open(os.path.join(DATA_DIR, pdf), 'rb') as f:
            hasher.update(f.read())
    
    return hasher.hexdigest()

def is_db_oudated():
    if not os.path.exists(METADATA_FILE):
        return True
    
    try:
        with open(METADATA_FILE, 'rb') as f:
            saved_hash = pickle.load(f).get("pdf_hash", None)

        current_hash = compute_pdf_hash()

        return saved_hash != current_hash
    except Exception as e:
        print(f"Error checking database status: {e}\n")
        return True
    
def build_collection():
    global index, chunks

    start_time = time.time()

    if not is_db_oudated():
        print("Database is up to date. No need to rebuild.\n")
        return
    
    print("Building collection...\n")

    loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader, use_multithreading=True)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.\n")

    exisitng_collections = [col.name for col in q_client.get_collections().collections]

    if COLLECTION_NAME in exisitng_collections:
        q_client.update_collection(
            collection_name=COLLECTION_NAME,
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=10000),
        )
    else:
        q_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            ),
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=10000),
        )

    vector_store.add_documents(chunks)

    with open(METADATA_FILE, 'wb') as f:
        pickle.dump({"pdf_hash": compute_pdf_hash()}, f)
    
    print(f"Collection {COLLECTION_NAME} built successfully in {time.time() - start_time:.2f} seconds.\n")

def generate_answer(query, k=5):
    results = vector_store.similarity_search(query, k=k)

    context = "\n\n".join([doc.page_content for doc in results])

    print(f"Context for query:\n{context}\n")

    prompt = f"use the provided context to asnwer the question {query} \n\n Context:\n{context}\n\nAnswer:"

    response = chat(
        model="llama3.2",
        messages=[
            {
                "role":"system",
                "content": "You're a helpful student assistant trying to answer queries using course material."
            },
            {
                "role":"user",
                "content": prompt

            }
        ]
    )

    return response.message.content.strip()

def main():
    build_collection()

    query = input("Enter your query \n")

    start_time = time.time()

    print("Processing query...\n")

    answer = generate_answer(query)
    print(f"Answer: {answer}\n")

    print(f"Query processed in {time.time() - start_time:.2f} seconds.\n")

if __name__ == "__main__":
    main()