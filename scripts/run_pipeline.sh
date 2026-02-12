#!/bin/bash

# Run ML Pipeline Script
# This script executes the complete ML pipeline end-to-end

set -e

echo "================================"
echo "ML Pipeline Execution"
echo "================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}Virtual environment not activated. Activating...${NC}"
    source venv/bin/activate
fi

# Step 1: Load and validate data
echo -e "\n${YELLOW}Step 1/4: Loading and validating data...${NC}"
python3 -m src.data.data_loader
python3 -m src.data.data_validator

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Data validation failed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Data loaded and validated${NC}"

# Step 2: Engineer features
echo -e "\n${YELLOW}Step 2/4: Engineering features...${NC}"
python3 -m src.features.feature_engineering

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Feature engineering failed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Features engineered${NC}"

# Step 3: Train models
echo -e "\n${YELLOW}Step 3/4: Training models...${NC}"
python3 -m src.models.train

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Model training failed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Models trained${NC}"

# Step 4: Generate reports
echo -e "\n${YELLOW}Step 4/4: Generating reports...${NC}"
echo -e "${GREEN}✓ Pipeline completed${NC}"

# Display results
echo -e "\n${GREEN}================================${NC}"
echo -e "${GREEN}Pipeline Execution Complete!${NC}"
echo -e "${GREEN}================================${NC}"

# Show MLflow UI command
echo -e "\nTo view experiment results:"
echo -e "  ${YELLOW}mlflow ui --backend-store-uri file:./mlflow/mlruns${NC}"
echo -e "\nTo view reports:"
echo -e "  ${YELLOW}ls -lh reports/figures/${NC}"
echo -e "\nTo start the API:"
echo -e "  ${YELLOW}./scripts/deploy_model.sh${NC}\n"
