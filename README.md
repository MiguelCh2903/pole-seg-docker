# pole-inference

Uso mínimo

Prerequisitos:
- Docker
- Modelo en `models/` (ej. `best_yolov8m_seg.onnx` para CPU o `best_yolov8m_seg.engine` para GPU)

Build:

```bash
./build.sh
# o manualmente
export DOCKER_BUILDKIT=1
docker compose build --parallel
```

Ejecutar (CPU):

```bash
docker compose --profile cpu up -d
```

Ejecutar (GPU):

```bash
docker compose --profile gpu up -d
```

Comprobar servicio:

```bash
curl http://localhost:8000/health
```

Archivos clave:
- `app/` - código
- `models/` - modelos
- `Dockerfile`, `docker-compose.yml`, `requirements.txt`

Fin.
    response = requests.post(
        'http://localhost:8000/predict/image',
        files={'file': open('test_image.jpg', 'rb')}
    )
    
    with open('result.jpg', 'wb') as out:
        out.write(response.content)
```

## 📦 Requisitos del Sistema

### CPU:
- Python 3.10+
- 2GB RAM mínimo
- Docker 20.10+

### GPU:
- NVIDIA GPU con CUDA 11.8+
- NVIDIA Docker runtime
- 4GB VRAM mínimo recomendado
- Drivers NVIDIA actualizados

## 🔧 Desarrollo Local (sin Docker)

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
export MODEL_PATH=./models/yolo_seg_model.onnx
export USE_GPU=false

# Ejecutar servidor
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 📊 Logs y Monitoreo

Ver logs del contenedor:

```bash
# CPU
docker-compose --profile cpu logs -f

# GPU
docker-compose --profile gpu logs -f
```

Los logs incluyen:
- Información del dispositivo (CPU/GPU)
- Tiempo de carga del modelo
- Número de detecciones por request
- Errores y warnings

## 🐛 Troubleshooting

### El modelo no se carga
```bash
# Verificar que el archivo existe
ls -lh models/

# Verificar permisos
chmod 644 models/yolo_seg_model.onnx
```

### GPU no se detecta
```bash
# Verificar NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Verificar drivers
nvidia-smi
```
