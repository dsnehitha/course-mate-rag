# CourseMate RAG Application

A modular, extensible RAG (Retrieval-Augmented Generation) application for processing and querying course materials with support for multimodal content (text and images).

## 🚀 Features

- **Multimodal Processing**: Extract and process both text and images from PDF documents
- **Modular Architecture**: Clean, extensible codebase designed for microservices
- **Environment-based Configuration**: Support for development, staging, and production environments
- **RESTful API**: FastAPI-based API with comprehensive documentation
- **Vector Database**: Qdrant integration for efficient similarity search
- **Caching**: Intelligent caching for image captions and metadata
- **Multi-threading**: Parallel processing for image captioning

## 📁 Project Structure

```
course-mate-rag/
├── src/
│   ├── api/                 # API layer (FastAPI)
│   │   ├── app.py          # Main FastAPI application
│   │   ├── models.py       # Pydantic models
│   │   └── routes.py       # API routes
│   ├── config/             # Configuration management
│   │   └── settings.py     # Environment-based settings
│   ├── core/               # Core functionality
│   │   ├── document_processor.py  # PDF and text processing
│   │   ├── image_processor.py     # Image captioning
│   │   └── vector_store.py        # Vector database operations
│   ├── services/           # Business logic layer
│   │   └── rag_service.py  # Main RAG orchestration
│   └── utils/              # Utilities
│       └── logger.py       # Logging utilities
├── tests/                  # Test files
├── docs/                   # Documentation
├── experiments/            # Experimental implementations
│   └── legacy_implementations/  # Old implementations
├── main.py                 # CLI entry point
├── api_server.py           # API server entry point
└── requirements.txt        # Dependencies
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd course-mate-rag
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama** (for LLM integration):
   ```bash
   # Follow instructions at https://ollama.ai/
   ollama pull llama3.2
   ollama pull llama3.2-vision
   ```

5. **Windows Users - Install FFmpeg** (for video processing):
   ```bash
   # Option 1: Using Chocolatey (recommended)
   choco install ffmpeg
   
   # Option 2: Using winget
   winget install ffmpeg
   
   # Option 3: Manual installation
   # Download from https://ffmpeg.org/download.html
   # Extract to C:\ffmpeg and add C:\ffmpeg\bin to PATH
   ```

6. **Start Qdrant** (vector database):
   ```bash
   # Using Docker
   
   docker volume create qdrant_storage
   docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
   
   # Or install locally
   # Follow instructions at https://qdrant.tech/documentation/guides/installation/
   ```

## 🚀 Usage

### CLI Mode

Run the interactive CLI application:

```bash
python main.py
```

### API Mode

Start the REST API server:

```bash
python api_server.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/v1/health

### API Endpoints

- `POST /api/v1/query` - Process a query
- `GET /api/v1/collection/info` - Get collection information
- `POST /api/v1/collection/rebuild` - Rebuild the vector collection
- `DELETE /api/v1/collection` - Delete the collection
- `POST /api/v1/cache/clear` - Clear caches
- `GET /api/v1/health` - Health check

## ⚙️ Configuration

The application uses environment-based configuration. Set the `ENVIRONMENT` variable:

```bash
# Development (default)
export ENVIRONMENT=development

# Staging
export ENVIRONMENT=staging

# Production
export ENVIRONMENT=production
```

### Environment Variables

- `ENVIRONMENT`: Application environment (development/staging/production)
- `QDRANT_URL`: Qdrant server URL
- `COLLECTION_NAME`: Vector collection name
- `API_HOST`: API server host
- `API_PORT`: API server port
- `CUDA_AVAILABLE`: Enable CUDA for embeddings (true/false)

## 📊 Architecture

### Core Components

1. **Document Processor**: Handles PDF extraction and text processing
2. **Image Processor**: Manages image captioning using vision models
3. **Vector Store Manager**: Orchestrates Qdrant operations
4. **RAG Service**: Main orchestration layer
5. **API Layer**: FastAPI-based REST API

### Data Flow

1. **Document Processing**: PDFs → Text + Images → Chunks
2. **Image Captioning**: Images → Captions → Text chunks
3. **Vector Storage**: Chunks → Embeddings → Qdrant
4. **Query Processing**: Query → Similarity Search → Context → LLM → Answer

## 🔧 Development

### Adding New Features

1. **Core Modules**: Add to `src/core/`
2. **Services**: Add to `src/services/`
3. **API Endpoints**: Add to `src/api/routes.py`
4. **Configuration**: Update `src/config/settings.py`

### Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### Code Quality

```bash
# Format code
black src/

# Lint code
flake8 src/
```

## 🚀 Future Enhancements

### Phase 1: Project Structuring ✅
- [x] Modular codebase
- [x] Environment-based configuration
- [x] Clean architecture

### Phase 2: API Development ✅
- [x] RESTful API with FastAPI
- [x] Comprehensive documentation
- [x] Health checks and monitoring

### Phase 3: Multi-Course Implementation
- [ ] Course management system
- [ ] Batch processing for multiple courses
- [ ] Content versioning

### Phase 4: Enhanced Vector Database
- [ ] Multi-tenant Qdrant design
- [ ] Optimized collections for multiple courses

### Phase 5: Advanced Features
- [ ] Federated search across courses
- [ ] Feedback-based refinement
- [ ] Conversational memory
- [ ] Query suggestions
- [ ] Real-time content updates
- [ ] Redis caching
- [ ] Load balancing
- [ ] GPU clustering

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For support and questions, please open an issue on GitHub.
