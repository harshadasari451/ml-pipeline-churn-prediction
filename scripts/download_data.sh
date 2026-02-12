#!/bin/bash

# Download/Generate Data Script
# This script generates or downloads the churn prediction dataset

set -e

echo "================================"
echo "Data Download/Generation"
echo "================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}Virtual environment not activated. Activating...${NC}"
    source venv/bin/activate
fi

# Create data directories if they don't exist
echo -e "\n${YELLOW}Creating data directories...${NC}"
mkdir -p data/raw data/processed

# Generate synthetic data
echo -e "\n${YELLOW}Generating synthetic churn data...${NC}"
python3 -c "
from src.data.data_loader import DataLoader
from src.utils.logger import setup_logging

setup_logging()
loader = DataLoader()

# Generate and save data
df = loader.load_data()
print(f'Generated {len(df)} customer records')

# Split and save
train_df, val_df, test_df = loader.split_data(df)
loader.save_split_data(train_df, val_df, test_df)

print(f'Data split:')
print(f'  - Training: {len(train_df)} samples')
print(f'  - Validation: {len(val_df)} samples')
print(f'  - Test: {len(test_df)} samples')
print(f'Successfully saved data to data/ directory')
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Data generated successfully${NC}"
else
    echo -e "\n${YELLOW}⚠ Error generating data${NC}"
    exit 1
fi

# Display data statistics
echo -e "\n${YELLOW}Data Statistics:${NC}"
ls -lh data/raw/
ls -lh data/processed/

echo -e "\n${GREEN}================================${NC}"
echo -e "${GREEN}Data Download Complete!${NC}"
echo -e "${GREEN}================================${NC}"
echo -e "\nData files created:"
echo -e "  - data/raw/churn_data.csv"
echo -e "  - data/processed/train.pkl"
echo -e "  - data/processed/val.pkl"
echo -e "  - data/processed/test.pkl\n"
