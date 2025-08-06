"""
Configuration management for CourseMate RAG Application.
Supports different deployment scenarios (development, staging, production).
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from enum import Enum


class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "history_course_materials"
    vector_size: int = 384
    distance_metric: str = "COSINE"
    indexing_threshold: int = 10000


@dataclass
class ModelConfig:
    """Model configuration settings."""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "llama3.2"
    vision_model: str = "llama3.2-vision"
    device: str = "cuda" if os.environ.get("CUDA_AVAILABLE", "false").lower() == "true" else "cpu"
    max_workers: int = 5


@dataclass
class ProcessingConfig:
    """Document processing configuration."""
    chunk_size: int = 500
    chunk_overlap: int = 70
    similarity_search_k: int = 5
    data_dir: str = "./course_materials"
    image_dir: str = "./extracted_images"
    metadata_dir: str = "./course_materials/metadata"


@dataclass
class APIConfig:
    """API configuration settings."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False


class Settings:
    """Main settings class that manages all configuration."""
    
    def __init__(self, env: Optional[str] = None):
        self.environment = Environment(env or os.getenv("ENVIRONMENT", "development"))
        self._load_environment_specific_settings()
    
    def _load_environment_specific_settings(self):
        """Load environment-specific settings."""
        if self.environment == Environment.DEVELOPMENT:
            self._load_development_settings()
        elif self.environment == Environment.STAGING:
            self._load_staging_settings()
        elif self.environment == Environment.PRODUCTION:
            self._load_production_settings()
    
    def _load_development_settings(self):
        """Development environment settings."""
        self.database = DatabaseConfig()
        self.model = ModelConfig()
        self.processing = ProcessingConfig()
        self.api = APIConfig(debug=True, reload=True)
    
    def _load_staging_settings(self):
        """Staging environment settings."""
        self.database = DatabaseConfig(
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            collection_name=os.getenv("COLLECTION_NAME", "history_course_materials")
        )
        self.model = ModelConfig(
            device="cpu",  # Staging typically uses CPU
            max_workers=3
        )
        self.processing = ProcessingConfig()
        self.api = APIConfig(debug=False, reload=False)
    
    def _load_production_settings(self):
        """Production environment settings."""
        self.database = DatabaseConfig(
            qdrant_url=os.getenv("QDRANT_URL", "http://qdrant:6333"),
            collection_name=os.getenv("COLLECTION_NAME", "history_course_materials"),
            indexing_threshold=50000
        )
        self.model = ModelConfig(
            device="cuda",
            max_workers=10
        )
        self.processing = ProcessingConfig(
            chunk_size=1000,
            chunk_overlap=100
        )
        self.api = APIConfig(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            debug=False,
            reload=False
        )


# Global settings instance
settings = Settings() 