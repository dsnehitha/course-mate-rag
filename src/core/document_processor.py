"""
Document processing module for CourseMate RAG Application.
Handles PDF extraction, text processing, and image extraction.
"""

import os
import hashlib
import pickle
import time
import io
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import pymupdf
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config.settings import settings


class DocumentProcessor:
    """Handles document processing including PDF extraction and image processing."""
    
    def __init__(self, settings=None):
        self.settings = settings
        self.data_dir = Path(self.settings.processing.data_dir)
        self.image_dir = Path(self.settings.processing.image_dir)
        self.metadata_dir = Path(self.settings.processing.metadata_dir)
        self.metadata_file = self.metadata_dir / "metadata.pkl"
        
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.text_data = []
        self.image_data = []
    
    def compute_pdf_hash(self) -> str:
        """Compute hash of all PDF files in the data directory."""
        hasher = hashlib.sha256()
        pdf_files = sorted([f for f in self.data_dir.iterdir() if f.suffix.lower() == '.pdf'])
        
        for pdf in pdf_files:
            with open(pdf, 'rb') as f:
                hasher.update(f.read())
        
        return hasher.hexdigest()
    
    def is_database_outdated(self) -> bool:
        """Check if the database needs to be rebuilt based on content changes."""
        # Always check if metadata file exists
        if not self.metadata_file.exists():
            return True
        
        # Check if Qdrant collection exists
        try:
            from qdrant_client import QdrantClient
            from src.config.settings import settings
            
            client = QdrantClient(
                url=settings.database.qdrant_url,
                check_compatibility=False  # Suppress version compatibility warnings
            )
            existing_collections = [
                col.name for col in client.get_collections().collections
            ]
            
            if settings.database.collection_name not in existing_collections:
                print("Qdrant collection doesn't exist. Database needs to be rebuilt.")
                return True
                
        except Exception as e:
            print(f"Error checking Qdrant collection: {e}")
            return True
        
        try:
            with open(self.metadata_file, 'rb') as f:
                saved_metadata = pickle.load(f)
            
            # Check PDF hash
            saved_pdf_hash = saved_metadata.get("pdf_hash", None)
            current_pdf_hash = self.compute_pdf_hash()
            
            if saved_pdf_hash != current_pdf_hash:
                print("PDF content has changed. Database needs to be rebuilt.")
                return True
            
            # Check if we have audio/video tracking (for backward compatibility)
            if "audio_video_hash" not in saved_metadata:
                print("Audio/video tracking not found in metadata. Database needs to be rebuilt.")
                return True
            
            # Check audio/video hash
            saved_av_hash = saved_metadata.get("audio_video_hash", None)
            current_av_hash = self.compute_audio_video_hash()
            
            if saved_av_hash != current_av_hash:
                print("Audio/video content has changed. Database needs to be rebuilt.")
                return True
            
            print("Database is up to date. No need to rebuild.")
            return False
            
        except Exception as e:
            print(f"Error checking database status: {e}")
            return True
    
    def compute_audio_video_hash(self) -> str:
        """Compute hash of all audio and video files in the data directory."""
        hasher = hashlib.sha256()
        audio_video_files = []
        
        # Get supported audio/video formats
        audio_formats = ['.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg']
        video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        supported_formats = audio_formats + video_formats
        
        for file_path in self.data_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_formats:
                audio_video_files.append(file_path)
        
        # Sort files for consistent hashing
        audio_video_files.sort(key=lambda x: x.name)
        
        for file_path in audio_video_files:
            with open(file_path, 'rb') as f:
                hasher.update(f.read())
        
        return hasher.hexdigest()
    
    def extract_images_from_pdfs(self) -> None:
        """Extract text and images from all PDF files."""
        self.text_data = []
        self.image_data = []
        
        for file_path in self.data_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() == '.pdf':
                print(f"\nExtracting images from {file_path.name}...")
                
                start_time = time.time()
                
                with pymupdf.open(str(file_path)) as doc:
                    for page_num in range(len(doc)):
                        page = doc[page_num]
                        
                        # Extract text
                        text = page.get_text().strip()
                        self.text_data.append({
                            "response": text, 
                            "name": page_num + 1,
                            "source_file": file_path.name
                        })
                        
                        # Extract images
                        images = page.get_images(full=True)
                        
                        for img_index, img in enumerate(images, start=0):
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            image_ext = base_image["ext"]
                            
                            image_filename = (
                                self.image_dir / 
                                f"{file_path.stem}_page_{page_num+1}_img_{img_index}.{image_ext}"
                            )
                            
                            image = Image.open(io.BytesIO(image_bytes))
                            image.save(str(image_filename))
                
                print(f"Extracted text and images in {time.time() - start_time:.2f} seconds.")
    
    def create_documents(self) -> Tuple[List[Document], List[Document]]:
        """Create Document objects from extracted text and image data."""
        # Create text documents
        text_docs = [
            Document(
                page_content=text['response'], 
                metadata={
                    "page": text['name'],
                    "source_file": text['source_file'],
                    "type": "text"
                }
            ) 
            for text in self.text_data
        ]
        
        # Create image documents (these will be populated with captions later)
        image_docs = []
        
        
        return text_docs, image_docs
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks using the configured text splitter."""
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=settings.processing.chunk_size,
            chunk_overlap=settings.processing.chunk_overlap
        )
        return text_splitter.split_documents(documents)
    
    def save_metadata(self) -> None:
        """Save metadata including PDF and audio/video hashes for change detection."""
        metadata = {
            "pdf_hash": self.compute_pdf_hash(),
            "audio_video_hash": self.compute_audio_video_hash()
        }
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
    
    def process_documents(self) -> Tuple[List[Document], List[Document]]:
        """Main method to process all documents."""
        
        self.extract_images_from_pdfs()
        text_docs, image_docs = self.create_documents()
        
        # Split documents
        text_splits = self.split_documents(text_docs)
        image_splits = self.split_documents(image_docs)
        
        return text_splits, image_splits 