import os
import torch
import hashlib
import pickle
import time
import io
import re
import pymupdf
import concurrent.futures
from ollama import chat
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant
from qdrant_client.http import models

DATA_DIR = "./course_materials"
IMAGE_DIR = "./extracted_images"
METADATA_DIR = "./course_materials/metadata"
METADATA_FILE = os.path.join(METADATA_DIR, "metadata.pkl")
COLLECTION_NAME = "history_course_materials"
QDRANT_URL = "http://localhost:6333"

os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

index = None
chunks = None
text_data = []
image_data = []

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

def extract_images_from_pdfs():
    global text_data, image_data

    for file_name in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.pdf'):
            print(f"\nExtracting images from {file_name}...\n")

            start_time = time.time()

            with pymupdf.open(file_path) as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]

                    text = page.get_text().strip()
                    text_data.append({"response": text, "name": page_num+1})

                    images = page.get_images(full=True)

                    for img_index, img in enumerate(images, start=0):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]

                        image_filename = f"{IMAGE_DIR}/{file_name[:-4]}_page_{page_num+1}_img_{img_index}.{image_ext}"

                        image = Image.open(io.BytesIO(image_bytes))
                        image.save(image_filename)
            
    print(f"Extracted text and images in {time.time() - start_time:.2f} seconds.\n")

def caption_single_image(img_path, img_name):
    try:
        response = chat(
            model="llama3.2-vision",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates captions for images."
                },
                {
                    "role": "user",
                    "content": "You are an assistant tasked with summarizing tables, images and text NASA website for retrieval. \
                                These summaries will be embedded and used to retrieve the raw text or table elements. \
                                Give a concise summary of the table or text that is optimized for retrieval.",
                    "images": [img_path],
                }
            ],
        )
        formatted_response = f"<IMG src=extracted_images/{img_name}>" + response.message.content + "<IMG>"
        return {"response": formatted_response, "name": img_name}
    except Exception as e:
        print(f"Error captioning {img_name}: {e}")
        return None

def generate_image_captions():
    global text_data, image_data

    captions_file = os.path.join(METADATA_DIR, "captions.pkl")

    if os.path.exists(captions_file):
        print("\nLoading cached image captions...\n")
        with open(captions_file, 'rb') as f:
            image_data = pickle.load(f)
    else:
        print("\nGenerating image captions in parallel...\n")

        start_time = time.time()
        img_files = [img for img in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, img))]

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for img in img_files:
                img_path = os.path.join(IMAGE_DIR, img)
                futures.append(executor.submit(caption_single_image, img_path, img))

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    image_data.append(result)

        with open(captions_file, 'wb') as f:
            pickle.dump(image_data, f)

    doc_list = [Document(page_content=text['response'], metadata={"page": text['name']}) for text in text_data]

    img_list = []

    for img in image_data:
        img_name = img['name']
        page_match = re.search(r'page_(\d+)_img', img_name)
        if page_match:
            page_num = int(page_match.group(1))
        else:
            page_num = None

        img_list.append(Document(
            page_content=img['response'],
            metadata={"page": page_num, "image_name": img_name}
        ))

    print(f"Generated image captions in {time.time() - start_time:.2f} seconds.\n.\n")
    return doc_list, img_list

def build_collection():
    global index, chunks

    start_time = time.time()

    if not is_db_oudated():
        print("Database is up to date. No need to rebuild.\n")
        return
    
    extract_images_from_pdfs()
    
    doc_list, img_list = generate_image_captions()
    
    print("\nBuilding collection...\n")

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=70)

    doc_splits = text_splitter.split_documents(doc_list)
    img_splits = text_splitter.split_documents(img_list)
    
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
    
    documents = doc_splits + img_splits

    vector_store.add_documents(documents)

    with open(METADATA_FILE, 'wb') as f:
        pickle.dump({"pdf_hash": compute_pdf_hash()}, f)
    
    print(f"Collection {COLLECTION_NAME} built successfully in {time.time() - start_time:.2f} seconds.\n")

def generate_answer(query, k=5):
    results = vector_store.similarity_search(query, k=k)

    context_blocks = []
    source_refs = []
    image_paths = []

    for doc in results:
        text = doc.page_content
        meta = doc.metadata

        images = re.findall(r"<IMG\s+src=([^>]+)>", text)
        if images:
            image_paths.extend(images)

        clean_text = re.sub(r"<IMG\s+src=[^>]+>", "", text)

        page_info = meta.get('page', 'unknown')
        image_info = meta.get('image_name', None)

        if page_info is not None:
            ref = f"Page {page_info}"
            if image_info:
                ref += f", Image {image_info}"
            source_refs.append(ref)
            context_blocks.append(f"(Source: {ref})\n{clean_text}")
        else:
            source_refs.append("unknown")
            context_blocks.append(f"(Source: unknown)\n{clean_text}")

    clean_context = "\n\n".join(context_blocks)

    print(f"\nContext for query:\n{clean_context}\n")

    prompt = f"use the provided context to answer the question: {query}\n\nContext:\n{clean_context}\n\nAnswer:"

    try:
        response = chat(
            model="llama3.2",
            messages=[
                {
                    "role": "system",
                    "content": "You're a helpful student assistant trying to answer queries using course material."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        answer = response.message.content.strip()

        print(f"\nAnswer: {answer}\n")

        print(f"\nSources referenced in the context:\n")
        for ref in set(source_refs):
            print(f"- {ref}")

        if image_paths:
            print(f"\nImage(s) used in context:\n")
            for path in image_paths:
                try:
                    if os.path.isfile(path):
                        print(f"Showing image: {path}")
                        img = Image.open(path)
                        img.show()
                    else:
                        print(f"Image file not found: {path}")
                except Exception as e:
                    print(f"Error opening image {path}: {e}")

        return answer

    except Exception as e:
        print(f"Error generating answer: {e}")
        return "An error occurred while generating the answer."
    
def main():
    build_collection()

    query = input("\n\nEnter your query \n")

    start_time = time.time()

    print("\n\nProcessing query...\n")

    answer = generate_answer(query)
    print(f"\nAnswer: {answer}\n")

    print(f"Query processed in {time.time() - start_time:.2f} seconds.\n")

if __name__ == "__main__":
    main()