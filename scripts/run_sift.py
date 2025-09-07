import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))  # Ensure src is in path
from src.utils.sift import SIFT3D

def get_files_from_folder(folder_path):
    """Get all .nii.gz files from a folder"""
    if not os.path.exists(folder_path):
        return []
    
    files = []
    for file in os.listdir(folder_path):
        if file.endswith('.nii.gz'):
            files.append(os.path.join(folder_path, file))
    
    return sorted(files)

def main():
    # Define paths
    data_dir = "./data/img"
    output_dir = "./data/sift_output"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize SIFT3D
    sift = SIFT3D()
    
    # Process MR files
    mr_folder = os.path.join(data_dir, "mr")
    mr_files = get_files_from_folder(mr_folder)
    
    if mr_files:
        print(f"🔍 Processing {len(mr_files)} MR files...")
        mr_output_dir = os.path.join(output_dir, "mr")
        os.makedirs(mr_output_dir, exist_ok=True)
        sift.process_images(mr_files, mr_output_dir, preprocess=True)
    
    # Process synthetic US files
    synthetic_us_folder = os.path.join(data_dir, "synthetic_us")
    synthetic_us_files = get_files_from_folder(synthetic_us_folder)
    
    if synthetic_us_files:
        print(f"🔍 Processing {len(synthetic_us_files)} synthetic US files...")
        synthetic_us_output_dir = os.path.join(output_dir, "synthetic_us")
        os.makedirs(synthetic_us_output_dir, exist_ok=True)
        sift.process_images(synthetic_us_files, synthetic_us_output_dir, preprocess=True)
    
    print("✅ SIFT processing completed!")
    print(f"📁 Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
