#!/bin/bash
set -e

# Colores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🐳 Building optimized Docker images using Ultralytics official base images${NC}\n"

# Build CPU image
echo -e "${GREEN}Building CPU image...${NC}"
docker build --target cpu -t yolo-inference:cpu .

# Build GPU image
echo -e "\n${GREEN}Building GPU image...${NC}"
docker build --target gpu -t yolo-inference:gpu .

echo -e "\n${BLUE}✅ Build complete!${NC}"
echo -e "\n${GREEN}Images created:${NC}"
docker images | grep yolo-inference

echo -e "\n${BLUE}💡 Usage:${NC}"
echo -e "  CPU: docker compose --profile cpu up"
echo -e "  GPU: docker compose --profile gpu up"
