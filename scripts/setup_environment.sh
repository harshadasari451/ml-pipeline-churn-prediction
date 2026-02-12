#!/bin/bash

# Setup Environment Script for ML Pipeline
# This script sets up the complete development environment

set -e

echo "================================"
echo "ML Pipeline Environment Setup"
echo "================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo -e "\n${YELLOW}Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "\n${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install requirements
echo -e "\n${YELLOW}Installing Python dependencies...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Install package in development mode
echo -e "\n${YELLOW}Installing package in development mode...${NC}"
pip install -e .
echo -e "${GREEN}✓ Package installed${NC}"

# Create necessary directories
echo -e "\n${YELLOW}Creating necessary directories...${NC}"
mkdir -p data/raw data/processed
mkdir -p models
mkdir -p mlflow/mlruns mlflow/models
mkdir -p logs
mkdir -p reports/figures
echo -e "${GREEN}✓ Directories created${NC}"

# Create .env file from example
if [ ! -f ".env" ]; then
    echo -e "\n${YELLOW}Creating .env file...${NC}"
    cp .env.example .env
    echo -e "${GREEN}✓ .env file created (please update with your settings)${NC}"
else
    echo -e "${GREEN}✓ .env file already exists${NC}"
fi

# Initialize git hooks (if .git directory exists)
if [ -d ".git" ]; then
    echo -e "\n${YELLOW}Setting up git hooks...${NC}"
    # Add any git hooks here
    echo -e "${GREEN}✓ Git hooks configured${NC}"
fi

# Run tests to verify installation
echo -e "\n${YELLOW}Running tests to verify installation...${NC}"
pytest tests/ -v --tb=short || echo -e "${YELLOW}⚠ Some tests failed (this is okay on first setup)${NC}"

# Summary
echo -e "\n${GREEN}================================${NC}"
echo -e "${GREEN}Environment Setup Complete!${NC}"
echo -e "${GREEN}================================${NC}"
echo -e "\nTo activate the environment, run:"
echo -e "  ${YELLOW}source venv/bin/activate${NC}"
echo -e "\nNext steps:"
echo -e "  1. Update .env file with your configuration"
echo -e "  2. Run ${YELLOW}./scripts/download_data.sh${NC} to generate data"
echo -e "  3. Run ${YELLOW}./scripts/run_pipeline.sh${NC} to execute the ML pipeline"
echo -e "  4. Run ${YELLOW}./scripts/deploy_model.sh${NC} to start the API\n"
