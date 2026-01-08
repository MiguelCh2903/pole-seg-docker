# CPU stage (ONNX) - Usa la imagen oficial de Ultralytics CPU
FROM ultralytics/ultralytics:latest-cpu AS cpu

WORKDIR /app

# Instalar solo las dependencias adicionales que no vienen en la imagen de Ultralytics
# Ultralytics ya incluye: opencv-python, numpy, ultralytics
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    pydantic \
    onnx \
    onnxruntime

# Copiar código de la aplicación
COPY ./app /app/app
COPY ./models /app/models

ENV USE_GPU=false
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


# GPU stage (TensorRT) - Usa la imagen oficial de Ultralytics GPU
FROM ultralytics/ultralytics:latest AS gpu

WORKDIR /app

# Instalar TensorRT y dependencias adicionales
# Ultralytics GPU ya incluye: opencv-python, numpy, ultralytics, torch, torchvision
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    pydantic \
    tensorrt-cu12==10.14.1.48.post1 \
    onnx \
    onnxruntime-gpu

# Copiar código de la aplicación
COPY ./app /app/app
COPY ./models /app/models

ENV USE_GPU=true
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


# Default stage (CPU)
FROM cpu AS final
