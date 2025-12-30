# Base stage with common dependencies - usando Python 3.12
FROM python:3.12-slim as base

WORKDIR /app

# Instalar uv - gestor de paquetes ultra-rápido
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install system dependencies for OpenCV and ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# CPU stage (ONNX)
FROM base as cpu

# Instalar dependencias con uv (mucho más rápido que pip)
RUN uv pip install --system --break-system-packages --no-cache -r requirements.txt

COPY ./app /app/app
COPY ./models /app/models

ENV USE_GPU=false
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


# GPU stage (TensorRT) - CUDA 12 for TensorRT 10.x
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04 as gpu

WORKDIR /app

# Instalar uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install Python 3.12 (nativo en Ubuntu 24.04) and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

# Copy requirements
COPY requirements.txt .

# Instalar dependencias con uv (mucho más rápido que pip)
RUN uv pip install --system --break-system-packages --no-cache -r requirements.txt

COPY ./app /app/app
COPY ./models /app/models

ENV USE_GPU=true
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


# Default stage (CPU)
FROM cpu as final
