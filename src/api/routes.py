"""
FastAPI routes for CourseMate RAG Application.
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

from src.services.rag_service import RAGService
from src.api.models import (
    QueryRequest, QueryResponse, CollectionInfoResponse, 
    HealthResponse, RebuildRequest
)
from src.config.settings import settings

# Create router
router = APIRouter(prefix="/api/v1", tags=["rag"])

# Dependency to get RAG service
def get_rag_service():
    return RAGService()


@router.get("/health", response_model=HealthResponse)
async def health_check(rag_service: RAGService = Depends(get_rag_service)):
    """Health check endpoint."""
    try:
        collection_info = rag_service.get_collection_info()
        return HealthResponse(
            status="healthy",
            environment=settings.environment.value,
            collection_info=CollectionInfoResponse(**collection_info) if collection_info else None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Service unhealthy: {str(e)}")


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """Process a query and return an answer."""
    try:
        result = rag_service.generate_answer(request.query, request.k)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.get("/collection/info", response_model=CollectionInfoResponse)
async def get_collection_info(rag_service: RAGService = Depends(get_rag_service)):
    """Get information about the current collection."""
    try:
        info = rag_service.get_collection_info()
        if not info:
            raise HTTPException(status_code=404, detail="Collection not found")
        return CollectionInfoResponse(**info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting collection info: {str(e)}")


@router.post("/collection/rebuild")
async def rebuild_collection(
    request: RebuildRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """Rebuild the vector collection."""
    try:
        if request.clear_cache:
            rag_service.clear_cache()
        
        if request.force:
            rag_service.rebuild_collection()
        else:
            rag_service.build_collection()
        
        return JSONResponse(
            content={"message": "Collection rebuilt successfully"},
            status_code=200
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rebuilding collection: {str(e)}")


@router.delete("/collection")
async def delete_collection(rag_service: RAGService = Depends(get_rag_service)):
    """Delete the vector collection."""
    try:
        rag_service.vector_store.delete_collection()
        return JSONResponse(
            content={"message": "Collection deleted successfully"},
            status_code=200
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting collection: {str(e)}")


@router.post("/cache/clear")
async def clear_cache(rag_service: RAGService = Depends(get_rag_service)):
    """Clear all caches."""
    try:
        rag_service.clear_cache()
        return JSONResponse(
            content={"message": "Cache cleared successfully"},
            status_code=200
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}") 