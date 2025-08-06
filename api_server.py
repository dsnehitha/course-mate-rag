"""
API server entry point for CourseMate RAG Application.
"""

import uvicorn
from src.api.app import app
from src.config.settings import settings
from src.utils.logger import logger


def main():
    """Start the API server."""
    logger.info(f"Starting CourseMate RAG API server in {settings.environment.value} mode...")
    logger.info(f"Server will be available at http://{settings.api.host}:{settings.api.port}")
    logger.info(f"API documentation available at http://{settings.api.host}:{settings.api.port}/docs")
    
    uvicorn.run(
        "src.api.app:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        log_level="info" if settings.api.debug else "warning"
    )


if __name__ == "__main__":
    main() 