# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is an AI/ML MVP for Thrasio's data modernization project, built with Python and Poetry. The project integrates with Google Cloud Platform (specifically Vertex AI) and aims to consolidate legacy systems into a scalable BigQuery-based platform with AI/ML capabilities and Looker reporting.

## Development Environment Setup
This project uses Poetry for dependency management:

```bash
# Install dependencies
poetry install

# Activate virtual environment
poetry shell

# Install development dependencies (includes ruff, ipykernel, streamlit)
poetry install --with dev
```

## Essential Commands

### Testing
```bash
# Run tests (configured in pyproject.toml)
poetry run pytest

# Run tests with coverage
poetry run pytest --cov
```

### Code Quality
```bash
# Lint and format code (configured in pyproject.toml)
poetry run ruff check .

# Auto-fix linting issues
poetry run ruff check . --fix

# Format code
poetry run ruff format .
```

### Running the Application
```bash
# Run the main application (FastAPI/Flask app)
poetry run python app/app.py

# Run Streamlit applications (if any)
poetry run streamlit run <streamlit_app.py>

# Run Jupyter notebooks
poetry run jupyter notebook
```

### Pipeline Development
```bash
# Compile Vertex AI pipeline locally
poetry run python app/pipeline.py

# Deploy pipeline to Vertex AI
export PYTHONPATH=$PYTHONPATH:.
poetry run python app/deploy.py

# Train AutoML model
poetry run python app/automl_train.py
```

## Project Architecture
- **app/**: Main application code including:
  - `pipeline.py`: Vertex AI pipeline definition with KFP components
  - `deploy.py`: Deployment utilities for CI/CD pipeline compilation and submission
  - `automl_train.py`: AutoML model training and evaluation utilities
- **experiments/**: ML experiments and research code (excluded from ruff linting)
- **evaluations/**: ML model evaluation code (included in pytest path)
- **.github/workflows/**: CI/CD pipeline configuration for automated deployment

## Environment Configuration
Required environment variables (see .env.example):
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to GCP service account key
- `PROJECT_ID`: GCP project ID
- `SERVICE_ACCOUNT_EMAIL`: GCP service account email
- `REGION`: GCP region (e.g., us-central1)
- `MODEL_ID`: Gemini model ID (e.g., gemini-1.5-flash-002)

Place a `key.json` file (GCP service account key) in the project root.

## Key Dependencies
- **FastAPI/Flask**: Web framework
- **Vertex AI**: Google Cloud AI platform integration
- **OpenCV**: Computer vision processing
- **Pandas/NumPy**: Data processing
- **Pydantic**: Data validation
- **Instructor**: Structured LLM outputs