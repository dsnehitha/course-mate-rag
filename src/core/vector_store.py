"""
Vector store management module for CourseMate RAG Application.
Handles Qdrant operations and vector store management.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant
from qdrant_client.http import models

from src.config.settings import settings


class VectorStoreManager:
    """Manages vector store operations with Qdrant."""
    
    def __init__(self, settings=None):
        self.settings = settings
        self.client = QdrantClient(
            url=self.settings.database.qdrant_url, 
            prefer_grpc=False,
            check_compatibility=False
        )
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.settings.model.embedding_model,
            model_kwargs={"device": self.settings.model.device}
        )
        self.vector_store = Qdrant(
            client=self.client,
            collection_name=self.settings.database.collection_name,
            embeddings=self.embedding_model
        )
    
    def create_collection(self) -> None:
        """Create the vector collection if it doesn't exist."""
        existing_collections = [
            col.name for col in self.client.get_collections().collections
        ]
        
        if self.settings.database.collection_name not in existing_collections:
            self.client.create_collection(
                collection_name=self.settings.database.collection_name,
                vectors_config=models.VectorParams(
                    size=self.settings.database.vector_size,
                    distance=getattr(models.Distance, self.settings.database.distance_metric)
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=self.settings.database.indexing_threshold
                ),
            )
            print(f"Created collection: {self.settings.database.collection_name}")
        else:
            # Update existing collection
            self.client.update_collection(
                collection_name=self.settings.database.collection_name,
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=self.settings.database.indexing_threshold
                ),
            )
            print(f"Updated collection: {self.settings.database.collection_name}")
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        if documents:
            self.vector_store.add_documents(documents)
            print(f"Added {len(documents)} documents to vector store.")
    
    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Perform similarity search on the vector store."""
        k = k or self.settings.processing.similarity_search_k
        return self.vector_store.similarity_search(query, k=k)
    
    def delete_collection(self) -> None:
        """Delete the vector collection."""
        try:
            self.client.delete_collection(self.settings.database.collection_name)
            print(f"Deleted collection: {self.settings.database.collection_name}")
        except Exception as e:
            print(f"Error deleting collection: {e}")
    
    def get_collection_info(self) -> dict:
        """Get information about the current collection."""
        try:
            # Check if collection exists first
            existing_collections = [
                col.name for col in self.client.get_collections().collections
            ]
            if self.settings.database.collection_name not in existing_collections:
                return {"error": "Collection does not exist"}
            collection_info = self.client.get_collection(self.settings.database.collection_name)
            try:
                info_dict = collection_info.model_dump()
            except AttributeError:
                info_dict = collection_info.dict()
            return {
                "name": self.settings.database.collection_name,  # Use the collection name from settings
                "vectors_count": info_dict.get("vectors_count", 0),
                "indexed_vectors_count": info_dict.get("indexed_vectors_count", 0),
                "points_count": info_dict.get("points_count", 0),
                "status": str(info_dict.get("status", "Unknown")),
                "segments_count": info_dict.get("segments_count", 0)
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {"error": str(e)} 