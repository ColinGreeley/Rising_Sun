# ---------- stage 1: frontend build ----------
FROM node:24-slim AS frontend-build
WORKDIR /app/frontend
COPY web/frontend/package.json web/frontend/package-lock.json ./
RUN npm ci
COPY web/frontend/ ./
ARG VITE_API_BASE_URL=""
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
RUN npm run build

# ---------- stage 2: backend ----------
FROM python:3.11-slim AS backend

# System deps for OpenCV, PyMuPDF, lxml
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libxml2 libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps — CPU-only PyTorch first (keeps image smaller)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir ".[handwriting]" \
    fastapi uvicorn httpx lxml

# Copy backend code
COPY web/backend/ ./web/backend/

# Copy TrOCR model weights (exclude training checkpoints)
COPY output/trocr_model_v3b/config.json \
     output/trocr_model_v3b/generation_config.json \
     output/trocr_model_v3b/model.safetensors \
     output/trocr_model_v3b/processor_config.json \
     output/trocr_model_v3b/tokenizer.json \
     output/trocr_model_v3b/tokenizer_config.json \
     ./output/trocr_model_v3b/

# Copy built frontend into backend's static directory
COPY --from=frontend-build /app/frontend/dist ./web/frontend/dist

# IDOC spreadsheet is optional — app degrades gracefully without it
# COPY IDOC/Data/1.\ Processed\ Apps\ List.xlsx ./IDOC/Data/

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--app-dir", "/app/web/backend"]
