.PHONY: help install test lint format clean run-api run-cli docker-build docker-run docker-stop

help: ## Show this help message
	@echo "CourseMate RAG Application"
	@echo "=========================="
	@echo ""
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install -e .[dev]

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

lint: ## Run linting
	flake8 src/ tests/
	mypy src/

format: ## Format code
	black src/ tests/
	isort src/ tests/

clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/

run-api: ## Run the API server
	python api_server.py

run-cli: ## Run the CLI application
	python main.py

docker-build: ## Build Docker image
	docker build -t coursemate-rag .

docker-run: ## Run with Docker Compose
	docker-compose up -d

docker-stop: ## Stop Docker Compose services
	docker-compose down

docker-logs: ## View Docker logs
	docker-compose logs -f

setup-dev: ## Setup development environment
	python -m venv venv
	@echo "Virtual environment created. Activate it with:"
	@echo "  source venv/bin/activate  # On Unix/Mac"
	@echo "  venv\\Scripts\\activate     # On Windows"
	@echo ""
	@echo "Then run: make install-dev"

rebuild-collection: ## Rebuild the vector collection
	curl -X POST "http://localhost:8000/api/v1/collection/rebuild" \
		-H "Content-Type: application/json" \
		-d '{"force": true, "clear_cache": true}'

health-check: ## Check API health
	curl -f http://localhost:8000/api/v1/health

collection-info: ## Get collection information
	curl -f http://localhost:8000/api/v1/collection/info 