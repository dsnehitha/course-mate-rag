"""
RAG service module for CourseMate RAG Application.
Orchestrates document processing, vector storage, and query answering.
"""

import time
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from ollama import chat
from langchain_core.documents import Document

from src.core.document_processor import DocumentProcessor
from src.core.image_processor import ImageProcessor
from src.core.audio_video_processor import AudioVideoProcessor
from src.core.vector_store import VectorStoreManager
from src.config.settings import settings


class RAGService:
    """Main RAG service that orchestrates all operations."""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.image_processor = ImageProcessor()
        self.audio_video_processor = AudioVideoProcessor()
        self.vector_store = VectorStoreManager()
    
    def build_collection(self) -> None:
        """Build the vector collection from documents."""
        # Check if database needs to be rebuilt
        if not self.document_processor.is_database_outdated():
            print("Database is up to date. No need to rebuild.")
            return
        
        start_time = time.time()
        print("Database is outdated. Rebuilding collection...")
        
        # Process documents
        text_splits, image_splits = self.document_processor.process_documents()
        
        # Process audio and video files
        audio_splits, video_splits = self.audio_video_processor.get_processed_files()
        
        if not text_splits and not image_splits and not audio_splits and not video_splits:
            print("No documents to process.")
            return
        
        # Generate image captions
        if image_splits:
            image_docs = self.image_processor.generate_image_captions()
            image_splits = self.document_processor.split_documents(image_docs)
        
        # Split audio and video documents
        if audio_splits:
            audio_splits = self.document_processor.split_documents(audio_splits)
        
        if video_splits:
            video_splits = self.document_processor.split_documents(video_splits)
        
        # Create vector store (only if needed)
        self.vector_store.create_collection()
        
        # Add documents to vector store
        all_documents = text_splits + image_splits + audio_splits + video_splits
        self.vector_store.add_documents(all_documents)
        
        # Save metadata
        self.document_processor.save_metadata()
        
        print(f"Collection built successfully in {time.time() - start_time:.2f} seconds.")
        
        self.print_database_status()
    
    def generate_answer(self, query: str, k: Optional[int] = None) -> Dict:
        """Generate answer for a query using RAG."""
        # Perform similarity search
        results = self.vector_store.similarity_search(query, k=k)
        
        # Process results
        context_blocks = []
        source_refs = []
        image_paths = []
        
        for doc in results:
            text = doc.page_content
            meta = doc.metadata
            
            # Extract image references
            images = re.findall(r"<IMG\s+src=([^>]+)>", text)
            if images:
                image_paths.extend(images)
            
            # Clean text
            clean_text = re.sub(r"<IMG\s+src=[^>]+>", "", text)
            
            # Build source reference
            page_info = meta.get('page', 'unknown')
            image_info = meta.get('image_name', None)
            source_file = meta.get('source_file', 'unknown')
            
            if page_info is not None:
                ref = f"Page {page_info}"
                if image_info:
                    ref += f", Image {image_info}"
                ref += f" ({source_file})"
            else:
                ref = f"Unknown source ({source_file})"
            
            source_refs.append(ref)
            context_blocks.append(f"(Source: {ref})\n{clean_text}")
        
        clean_context = "\n\n".join(context_blocks)
        
        system_prompt = f"""
        You are a helpful and knowledgeable student assistant. 
        Your job is to carefully answer student queries using the provided lecture or course material as the main source of truth. 
        Always ground your answers in the given context. If the context does not contain enough information, explicitly say so and avoid making things up. 
        Provide the answer only without prefixing it with statements such as "Based on the provided context" or "According to the lecture" or "Based on the provided transcript" etc.
        

        When answering:
        - Be clear, concise, and accurate. 
        - Use a structured explanation with definitions, step-by-step reasoning, and examples if applicable. 
        - Highlight key concepts or formulas from the context that directly support the answer. 
        - If the question is broad or open-ended, provide a summary first and then expand into details. 
        - If multiple interpretations are possible, explain them and clarify based on context.
        """
        
        # Generate answer using LLM
        prompt = f"Use the provided context to answer the question: {query}\n\nContext:\n{clean_context}\n\nAnswer:"
        
        try:
            response = chat(
                model=settings.model.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            answer = response.message.content.strip()
            
            return {
                "answer": answer,
                "context": clean_context,
                "sources": list(set(source_refs)),
                "images": list(set(image_paths)),
                "query": query
            }
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return {
                "answer": "An error occurred while generating the answer.",
                "context": clean_context,
                "sources": list(set(source_refs)),
                "images": list(set(image_paths)),
                "query": query,
                "error": str(e)
            }
    
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        self.image_processor.clear_cache()
        self.audio_video_processor.clear_cache()
        print("Cleared all caches.")
    
    def rebuild_collection(self) -> None:
        """Force rebuild the collection."""
        # Clear caches
        self.clear_cache()
        
        # Delete existing collection
        self.vector_store.delete_collection()
        
        # Rebuild
        self.build_collection() 
    
    def get_database_status(self) -> Dict:
        """Get detailed information about the current database status."""
        status = {
            "database_outdated": self.document_processor.is_database_outdated(),
            "collection_exists": False,
            "collection_info": {},
            "metadata_file_exists": self.document_processor.metadata_file.exists(),
            "content_hashes": {}
        }
        
        # Check if collection exists in Qdrant
        try:
            collection_info = self.vector_store.get_collection_info()
            status["collection_exists"] = bool(collection_info)
            status["collection_info"] = collection_info
        except Exception as e: 
            status["collection_info"] = {"error": str(e)}
        
        # Get content hashes if metadata exists
        if status["metadata_file_exists"]:
            try:
                import pickle
                with open(self.document_processor.metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
                status["content_hashes"] = {
                    "pdf_hash": metadata.get("pdf_hash", "Not found"),
                    "audio_video_hash": metadata.get("audio_video_hash", "Not found")
                }
            except Exception as e:
                status["content_hashes"] = {"error": str(e)}
        
        return status
    
    def print_database_status(self) -> None:
        """Print a human-readable database status."""
        status = self.get_database_status()
        
        print("\n=== Database Status ===")
        print(f"Database outdated: {'Yes' if status['database_outdated'] else 'No'}")
        print(f"Collection exists: {'Yes' if status['collection_exists'] else 'No'}")
        print(f"Metadata file exists: {'Yes' if status['metadata_file_exists'] else 'No'}")
        
        if status["database_outdated"]:
            print("\n⚠️  Database needs to be rebuilt!")
        else:
            print("\n✅ Database is up to date!")
        
        print("======================\n") 