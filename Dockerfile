# syntax=docker/dockerfile:1

FROM python:3.12-slim

# Install OS dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only dependency files first for cache efficiency
COPY pyproject.toml .
COPY README.md .

# Install pip tools
RUN python -m pip install --upgrade pip setuptools wheel

# Copy app code after dependency files (improves rebuild caching)
COPY . .

# Install project dependencies (explicit, avoid building wheel from workspace layout)
RUN pip install --no-cache-dir \
    fastapi>=0.135.1 \
    langchain>=1.2.12 \
    langchain-openai>=1.1.11 \
    pydantic>=2.12.5 \
    python-dotenv>=1.2.2 \
    spotipy>=2.25.1 \
    uvicorn>=0.42.0 \
    google-api-python-client>=2.70.0 \
    langchain-community>=0.4.1 \
    pymupdf>=1.27.2.2 \
    langchain-chroma>=1.1.0 \
    langchain-core>=1.2.23 \
    langgraph>=1.1.3 \
    langchain-tavily>=0.2.18

# Expose port for FastAPI
EXPOSE 8000

# Default command (uvicorn)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
