#!/bin/bash

set -e  # Exit on any error

echo "CrossKEY - Setup"
echo "=================================================="

# Check OS for SIFT3D support
OS="$(uname -s)"
if [ "$OS" != "Linux" ]; then
    echo "WARNING: SIFT3D compilation requires Linux (apt-get dependencies)."
    echo "On $OS, the Python environment will be set up but SIFT3D must be installed manually."
    echo "See: https://github.com/bbrister/SIFT3D"
    SKIP_SIFT3D=true
else
    SKIP_SIFT3D=false
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Please install uv first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "   Or visit: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

echo "uv found"

# Setup Python virtual environment with uv
echo "Setting up Python virtual environment with uv..."
uv venv .venv
uv pip install -e .

echo "Python dependencies installed"

# Install external libraries (Linux only)
if [ "$SKIP_SIFT3D" = false ]; then
    echo "Installing external libraries..."
    mkdir -p external_libs
    cd external_libs

    # Install SIFT3D
    echo "Installing SIFT3D..."
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

    echo "SIFT3D installed successfully"
else
    echo "Skipping SIFT3D installation (not on Linux)"
    echo "You will need to install SIFT3D manually for full functionality."
fi

# Setup data directories
echo "Setting up data directories..."
mkdir -p data/heatmap
mkdir -p data/sift_output/mr data/sift_output/synthetic_us
mkdir -p logs

echo "Data directories created"

echo ""
echo "Setup completed!"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Start training (automatically generates heatmaps and SIFT descriptors):"
echo "   python example_train.py"
echo ""
echo "3. Or test the installation first:"
echo "   python example_test.py"
echo ""
echo "Note: The training script will automatically run data preprocessing"
echo "      (SIFT extraction and heatmap generation) if needed."
echo ""
echo "For more information, see README.md"
