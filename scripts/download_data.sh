#!/bin/bash
# Script to download polyp segmentation datasets

set -e

echo "=========================================="
echo "Downloading Polyp Segmentation Datasets"
echo "=========================================="

# Create data directory
mkdir -p data
cd data

# Download Kvasir-SEG
echo ""
echo "Downloading Kvasir-SEG dataset..."
if [ ! -d "Kvasir-SEG" ]; then
    # Using gdown for Google Drive download
    pip install -q gdown
    
    # Kvasir-SEG Google Drive ID
    gdown https://drive.google.com/uc?id=1o8OfBvYE6K-EpDyvzsmMPndnUMwb540R
    
    # Unzip
    unzip -q TestDataset.zip
    rm TestDataset.zip
    
    echo "Kvasir-SEG downloaded successfully!"
    echo "Total images: $(ls TestDataset/images/*.jpg | wc -l)"
else
    echo "Kvasir-SEG already exists. Skipping..."
fi

# Download CVC-ClinicDB
echo ""
echo "Downloading CVC-ClinicDB dataset..."
if [ ! -d "CVC-ClinicDB" ]; then
    # Download from official source
    wget -q https://polyp.grand-challenge.org/CVCClinicDB/CVC-ClinicDB.zip
    
    # Unzip
    unzip -q CVC-ClinicDB.zip -d CVC-ClinicDB
    rm CVC-ClinicDB.zip
    
    echo "CVC-ClinicDB downloaded successfully!"
    echo "Total images: $(find CVC-ClinicDB/Original -type f | wc -l)"
else
    echo "CVC-ClinicDB already exists. Skipping..."
fi

cd ..

echo ""
echo "=========================================="
echo "Dataset Download Complete!"
echo "=========================================="
echo "Kvasir-SEG: data/Kvasir-SEG/"
echo "CVC-ClinicDB: data/CVC-ClinicDB/"
echo ""
echo "Dataset Structure:"
echo "data/"
echo "├── Kvasir-SEG/"
echo "│   ├── images/     (1000 polyp images)"
echo "│   └── masks/      (1000 segmentation masks)"
echo "└── CVC-ClinicDB/"
echo "    ├── Original/   (612 polyp images)"
echo "    └── Ground Truth/ (612 segmentation masks)"
echo ""
echo "You can now start training!"