"""
Pydantic models for API request/response schemas.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., description="The query to process")
    k: Optional[int] = Field(5, description="Number of similar documents to retrieve")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str = Field(..., description="Generated answer")
    context: str = Field(..., description="Context used for answer generation")
    sources: List[str] = Field(..., description="Source references")
    images: List[str] = Field(..., description="Image paths used in context")
    query: str = Field(..., description="Original query")


class CollectionInfoResponse(BaseModel):
    """Response model for collection info endpoint."""
    name: str = Field(..., description="Collection name")
    vectors_count: int = Field(..., description="Number of vectors")
    points_count: int = Field(..., description="Number of points")
    status: str = Field(..., description="Collection status")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Service status")
    environment: str = Field(..., description="Current environment")
    collection_info: Optional[CollectionInfoResponse] = Field(None, description="Collection information")


class RebuildRequest(BaseModel):
    """Request model for rebuild endpoint."""
    force: bool = Field(False, description="Force rebuild even if no changes detected")
    clear_cache: bool = Field(True, description="Clear cache before rebuild") 