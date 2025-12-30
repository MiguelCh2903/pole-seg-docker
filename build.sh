#!/bin/bash

# Script optimizado para buildear imágenes Docker con cache

set -e

echo "🚀 Building Docker images with BuildKit optimizations..."

# Habilitar BuildKit para builds más rápidos
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Build con caché y paralelización
docker compose --profile cpu build
docker compose --profile gpu build

echo "✅ Build completed successfully!"
echo ""
echo "Para iniciar los servicios:"
echo "  CPU:  docker compose --profile cpu up -d"
echo "  GPU:  docker compose --profile gpu up -d"
