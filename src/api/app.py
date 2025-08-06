"""
Main FastAPI application for CourseMate RAG Application.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.config.settings import settings

# Create FastAPI app
app = FastAPI(
    title="CourseMate RAG App",
    description="A RAG application for course material processing and querying",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "CourseMate RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    } 