"""
Image processing module for CourseMate RAG Application.
Handles image captioning and processing using vision models.
"""

import os
import pickle
import time
import re
from typing import List, Dict, Optional
from pathlib import Path
import concurrent.futures

from PIL import Image
from ollama import chat
from langchain_core.documents import Document

from src.config.settings import settings


class ImageProcessor:
    """Handles image processing and captioning."""
    
    def __init__(self):
        self.image_dir = Path(settings.processing.image_dir)
        self.metadata_dir = Path(settings.processing.metadata_dir)
        self.captions_file = self.metadata_dir / "captions.pkl"
        self.image_data = []
    
    def caption_single_image(self, img_path: Path, img_name: str) -> Optional[Dict]:
        """Generate caption for a single image using vision model."""
        try:
            response = chat(
                model=settings.model.vision_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates captions for images."
                    },
                    {
                        "role": "user",
                        "content": (
                            "You are an assistant tasked with summarizing tables, images and text for retrieval. These summaries will be embedded and used to retrieve the raw text or table elements. Give a concise summary of the table or text that is optimized for retrieval."
                        ),
                        "images": [str(img_path)],
                    }
                ],
            )
            
            formatted_response = f"<IMG src={img_path}>" + response.message.content + "<IMG>"
            return {"response": formatted_response, "name": img_name}
        except Exception as e:
            print(f"Error captioning {img_name}: {e}")
            return None
    
    def generate_image_captions(self) -> List[Document]:
        """Generate captions for all images in parallel."""
        if self.captions_file.exists():
            print("\nLoading cached image captions...")
            with open(self.captions_file, 'rb') as f:
                self.image_data = pickle.load(f)
        else:
            print("\nGenerating image captions in parallel...")
            
            start_time = time.time()
            img_files = [
                img for img in self.image_dir.iterdir() 
                if img.is_file() and img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
            ]
            
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=settings.model.max_workers
            ) as executor:
                futures = []
                for img_path in img_files:
                    futures.append(
                        executor.submit(self.caption_single_image, img_path, img_path.name)
                    )
                
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result:
                        self.image_data.append(result)
            
            # Cache the captions
            with open(self.captions_file, 'wb') as f:
                pickle.dump(self.image_data, f)
            
            print(f"Generated image captions in {time.time() - start_time:.2f} seconds.")
        
        # Convert to Document objects
        img_docs = []
        for img in self.image_data:
            img_name = img['name']
            page_match = re.search(r'page_(\d+)_img', img_name)
            page_num = int(page_match.group(1)) if page_match else None
            
            img_docs.append(Document(
                page_content=img['response'],
                metadata={
                    "page": page_num, 
                    "image_name": img_name,
                    "type": "image"
                }
            ))
        
        return img_docs
    
    def clear_cache(self) -> None:
        """Clear cached image captions."""
        if self.captions_file.exists():
            self.captions_file.unlink()
            print("Cleared image caption cache.") 