#!/bin/bash

set -e  # Exit on any error

echo "🔑 CrossKEY - A framework for learning 3D Cross-modal Keypoint Descriptor Setup"
echo "=================================================="

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry is not installed. Please install Poetry first:"
    echo "   curl -sSL https://install.python-poetry.org | python3 -"
    echo "   Or visit: https://python-poetry.org/docs/#installation"
    exit 1
fi

echo "✅ Poetry found"

# Setup Python virtual environment with Poetry
echo "📦 Setting up Python virtual environment with Poetry..."
poetry config virtualenvs.in-project true  # Create .venv in project directory
poetry install

echo "✅ Python dependencies installed"

# Install external libraries
echo "🔧 Installing external libraries..."
mkdir -p external_libs
cd external_libs

# Install SIFT3D
echo "📥 Installing SIFT3D..."
echo "Installing system dependencies for SIFT3D..."
sudo apt-get update && sudo apt-get install -y \
    zlib1g-dev \
    liblapack-dev \
    libdcmtk-dev \
    libnifti-dev \
    libblas-dev \
    cmake \
    build-essential

if [ ! -d "SIFT3D" ]; then
    echo "Cloning SIFT3D repository..."
    git clone https://github.com/morozovdd/SIFT3D.git
fi

cd SIFT3D
if [ ! -d "build" ]; then
    mkdir build
fi
cd build
echo "Building SIFT3D..."
cmake ..
make -j$(nproc)
cd ../..
cd ..

echo "✅ SIFT3D installed successfully"

# Setup data directories
echo "📁 Setting up data directories..."

# Create output directories (input data/img is included in repository)
mkdir -p data/heatmap
mkdir -p data/sift_output/mr data/sift_output/synthetic_us
mkdir -p logs

echo "✅ Data directories created"
echo "✅ Test data is included in the repository (data/img/)"

echo ""
echo "🎉 Setup completed successfully!"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   poetry shell"
echo ""
echo "2. Start training (automatically generates heatmaps and SIFT descriptors):"
echo "   poetry run python example_train.py"
echo ""
echo "3. Or test the installation first:"
echo "   poetry run python example_test.py"
echo ""
echo "Note: The training script will automatically run data preprocessing"
echo "      (SIFT extraction and heatmap generation) if needed."
echo ""
echo "For more information, see README.md"