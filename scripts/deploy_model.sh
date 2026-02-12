#!/bin/bash

# Deploy Model Script
# This script starts the FastAPI service and MLflow tracking server

set -e

echo "================================"
echo "Model Deployment"
echo "================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# Check deployment method
DEPLOY_METHOD=${1:-"local"}

if [ "$DEPLOY_METHOD" == "docker" ]; then
    echo -e "\n${YELLOW}Deploying with Docker Compose...${NC}"
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        echo "Error: Docker is not installed"
        exit 1
    fi
    
    # Build and start containers
    echo -e "${YELLOW}Building Docker images...${NC}"
    docker-compose build
    
    echo -e "${YELLOW}Starting containers...${NC}"
    docker-compose up -d
    
    echo -e "${GREEN}✓ Containers started${NC}"
    
    # Wait for services to be ready
    echo -e "\n${YELLOW}Waiting for services to be ready...${NC}"
    sleep 10
    
    # Check service health
    echo -e "${YELLOW}Checking service health...${NC}"
    curl -f http://localhost:8000/health || echo "API not ready yet"
    curl -f http://localhost:5000/health || echo "MLflow not ready yet"
    
    echo -e "\n${GREEN}================================${NC}"
    echo -e "${GREEN}Deployment Complete!${NC}"
    echo -e "${GREEN}================================${NC}"
    echo -e "\nServices running:"
    echo -e "  - API: ${YELLOW}http://localhost:8000${NC}"
    echo -e "  - API Docs: ${YELLOW}http://localhost:8000/docs${NC}"
    echo -e "  - MLflow UI: ${YELLOW}http://localhost:5000${NC}"
    echo -e "\nTo stop services:"
    echo -e "  ${YELLOW}docker-compose down${NC}"
    echo -e "\nTo view logs:"
    echo -e "  ${YELLOW}docker-compose logs -f${NC}\n"
    
else
    echo -e "\n${YELLOW}Deploying locally...${NC}"
    
    # Check if virtual environment is activated
    if [ -z "$VIRTUAL_ENV" ]; then
        echo -e "${YELLOW}Virtual environment not activated. Activating...${NC}"
        source venv/bin/activate
    fi
    
    # Start MLflow server in background
    echo -e "${YELLOW}Starting MLflow tracking server...${NC}"
    mlflow server \
        --backend-store-uri file:./mlflow/mlruns \
        --default-artifact-root ./mlflow/artifacts \
        --host 0.0.0.0 \
        --port 5000 &
    MLFLOW_PID=$!
    echo "MLflow PID: $MLFLOW_PID"
    
    # Wait a bit for MLflow to start
    sleep 5
    
    # Start FastAPI application
    echo -e "${YELLOW}Starting FastAPI application...${NC}"
    python3 -m uvicorn src.api.main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --reload &
    API_PID=$!
    echo "API PID: $API_PID"
    
    # Wait for services
    sleep 5
    
    echo -e "\n${GREEN}================================${NC}"
    echo -e "${GREEN}Deployment Complete!${NC}"
    echo -e "${GREEN}================================${NC}"
    echo -e "\nServices running:"
    echo -e "  - API: ${YELLOW}http://localhost:8000${NC}"
    echo -e "  - API Docs: ${YELLOW}http://localhost:8000/docs${NC}"
    echo -e "  - MLflow UI: ${YELLOW}http://localhost:5000${NC}"
    echo -e "\nPIDs:"
    echo -e "  - MLflow: $MLFLOW_PID"
    echo -e "  - API: $API_PID"
    echo -e "\nTo stop services:"
    echo -e "  ${YELLOW}kill $MLFLOW_PID $API_PID${NC}"
    echo -e "\nPress Ctrl+C to stop services\n"
    
    # Wait for user to stop
    wait
fi
