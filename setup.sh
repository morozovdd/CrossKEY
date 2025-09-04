#!/bin/bash


# Install external libraries
mkdir -p external_libs
cd external_libs
echo "Installing external libraries..."

# Install SIFT3D
echo "Installing SIFT3D..."
sudo apt-get update && sudo apt-get install -y zlib1g-dev liblapack-dev libdcmtk-dev libnifti-dev -y libblas-dev
git clone https://github.com/morozovdd/SIFT3D.git
cd SIFT3D
mkdir build
cd build
cmake ..
make
cd ..
cd ..

# Install MMHVAE
echo "Installing MMHVAE..."

cd ..

# Create data folder with proper structure
mkdir -p data/img/mr && mkdir -p data/img/us && mkdir -p data/img/synthetic_us
mkdir -p data/heatmap
mkdir -p data/sift_output

echo "Setup completed successfully!"