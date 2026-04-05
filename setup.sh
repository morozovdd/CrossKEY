#!/bin/bash

set -e  # Exit on any error

echo "CrossKEY - Setup"
echo "=================================================="

OS="$(uname -s)"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Please install uv first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "uv found"

# Setup Python virtual environment with uv
echo "Setting up Python virtual environment..."
uv venv .venv
uv pip install -e .
echo "Python dependencies installed"

# Install external libraries
echo "Installing SIFT3D..."
mkdir -p external_libs
cd external_libs

# Install system dependencies based on OS
if [ "$OS" = "Linux" ]; then
    echo "Installing system dependencies (Linux)..."
    sudo apt-get update && sudo apt-get install -y \
        zlib1g-dev \
        liblapack-dev \
        libdcmtk-dev \
        libnifti-dev \
        libblas-dev \
        cmake \
        build-essential
elif [ "$OS" = "Darwin" ]; then
    echo "Installing system dependencies (macOS)..."
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Install it from https://brew.sh"
        exit 1
    fi
    brew install cmake 2>/dev/null || true

    # Build nifticlib from source (not in Homebrew)
    if [ ! -d "nifti_clib" ]; then
        echo "Building nifticlib from source..."
        git clone https://github.com/NIFTI-Imaging/nifti_clib.git
    fi
    cd nifti_clib
    mkdir -p build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX="$(pwd)/../../nifti_install"
    make -j$(sysctl -n hw.ncpu)
    make install
    cd ../..
    NIFTI_PREFIX="$(pwd)/nifti_install"
    echo "nifticlib installed to $NIFTI_PREFIX"
else
    echo "Unsupported OS: $OS. Only Linux and macOS are supported."
    exit 1
fi

# Clone and build SIFT3D
if [ ! -d "SIFT3D" ]; then
    echo "Cloning SIFT3D repository..."
    git clone https://github.com/morozovdd/SIFT3D.git
fi

cd SIFT3D
mkdir -p build && cd build
echo "Building SIFT3D..."

if [ "$OS" = "Darwin" ]; then
    cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_PREFIX_PATH="$NIFTI_PREFIX"
    make -j$(sysctl -n hw.ncpu)
else
    cmake ..
    make -j$(nproc)
fi

cd ../..
cd ..

echo "SIFT3D installed successfully"

# Setup data directories
echo "Setting up data directories..."
mkdir -p data/heatmap
mkdir -p data/sift_output/mr data/sift_output/synthetic_us
mkdir -p logs

echo ""
echo "Setup completed!"
echo "================================"
echo ""
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  python example_train.py"
echo ""
echo "See README.md for details."
