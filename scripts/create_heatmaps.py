import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter, center_of_mass
from typing import List, Optional
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))  # Ensure src is in path
from src.utils.utils import load_nifti, save_nifti
import logging

logger = logging.getLogger("crosskey.heatmap")

class HeatmapProcessor:
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.sift_output_dir = self.data_dir.parent / "sift_output"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reference_nifti_path = None

    def load_keypoints(self, file_path: str) -> np.ndarray:
        """Load keypoints from CSV file."""
        data = pd.read_csv(file_path, header=None, delimiter=',')
        coordinates = data.iloc[:, :3].values
        return coordinates
    
    def get_sift_files_for_modality(self, modality_folder: str) -> List[Path]:
        """Get list of SIFT descriptor files for a specific modality from subfolder."""
        sift_modality_path = self.sift_output_dir / modality_folder
        if not sift_modality_path.exists():
            return []
        
        # List all *_desc.csv files in the SIFT output subfolder
        sift_files = list(sift_modality_path.glob("*_desc.csv"))
        return sorted(sift_files)
    
    def create_frequency_heatmap(self, volume_shape: tuple, modality_folder: str, sigma: float = 2.0) -> np.ndarray:
        """Create frequency heatmap for a specific modality based on subfolder structure."""
        frequency_map = np.zeros(volume_shape)
        
        # Get all SIFT descriptor files for this modality
        sift_files = self.get_sift_files_for_modality(modality_folder)
        
        if not sift_files:
            return frequency_map
        
        # Process each SIFT descriptor file with progress bar
        total_points = 0
        for sift_file in tqdm(sift_files, desc=f"Processing {modality_folder}", unit="file"):
            points = self.load_keypoints(str(sift_file))
            points = np.floor(points).astype(int)
            
            # Filter points within volume bounds
            mask = np.all((points >= 0) & (points < np.array(volume_shape)), axis=1)
            valid_points = points[mask]
            total_points += len(valid_points)
            
            # Add points to frequency map
            # SIFT coordinates are (x, y, z) in original NIfTI [X, Y, Z] format
            # Our volume is transposed to [H, W, D] = [Y, X, Z] format
            # So we map: SIFT x -> W (dim 1), SIFT y -> H (dim 0), SIFT z -> D (dim 2)
            for point in valid_points:
                x, y, z = point
                frequency_map[y, x, z] += 1
        
        logger.info("  %d files processed, %d keypoints added", len(sift_files), total_points)
        
        # Normalize and smooth the frequency map
        freq_normalized = self.normalize_map(frequency_map)
        smoothed_map = gaussian_filter(freq_normalized, sigma=sigma)
        
        return smoothed_map
    
    @staticmethod
    def normalize_map(heatmap: np.ndarray) -> np.ndarray:
        """Normalize heatmap to [0,1] range."""
        if heatmap.max() == heatmap.min():
            return heatmap
        return (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    @staticmethod
    def combine_heatmaps(heatmap1: np.ndarray, heatmap2: np.ndarray, method: str = 'or') -> np.ndarray:
        """Combine two heatmaps using specified method."""
        if heatmap1.shape != heatmap2.shape:
            raise ValueError(f"Heatmap shapes don't match: {heatmap1.shape} vs {heatmap2.shape}")
        
        h1_norm = HeatmapProcessor.normalize_map(heatmap1)
        h2_norm = HeatmapProcessor.normalize_map(heatmap2)
        
        if method == 'and':
            combined = h1_norm * h2_norm
        else:  # 'or'
            combined = h1_norm + h2_norm - h1_norm * h2_norm
            
        return HeatmapProcessor.normalize_map(combined)
    
    def create_weighted_fov_mask(self, binary_mask: np.ndarray, sigma_factor: float = 0.3) -> np.ndarray:
        """Create weighted FOV mask using Gaussian falloff."""
        h, w, d = binary_mask.shape
        weights = np.zeros_like(binary_mask, dtype=float)
        
        for d_idx in range(d):
            slice_mask = binary_mask[:, :, d_idx]
            if np.sum(slice_mask) > 0:
                # Get center of mass for this slice
                h_center, w_center = center_of_mass(slice_mask)
                
                # Create distance map for this slice
                h_coords, w_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
                dist = np.sqrt((h_coords - h_center)**2 + (w_coords - w_center)**2)
                
                # Calculate sigma based on the size of the FOV in this slice
                fov_points = np.array(np.where(slice_mask > 0))
                if fov_points.size > 0:
                    max_dist = np.max(np.sqrt(np.sum((fov_points - np.array([[h_center], [w_center]]))**2, axis=0)))
                    sigma = max_dist * sigma_factor
                    sigma = max(sigma, 1e-10)
                    
                    # Create Gaussian weights for this slice
                    slice_weights = np.exp(-0.5 * (dist/sigma)**2)
                    weights[:, :, d_idx] = slice_weights * slice_mask
        
        # Normalize weights to [0.1, 1.0] range where mask is non-zero
        mask_nonzero = binary_mask > 0
        if np.any(mask_nonzero):
            weights[mask_nonzero] = 0.1 + 0.9 * self.normalize_map(weights[mask_nonzero])
        
        return weights
    
    def get_reference_volume_shape(self) -> tuple:
        """Get volume shape from the MR file and store reference path."""
        # Try MR folder first
        mr_files = list((self.data_dir / "mr").glob("*.nii.gz"))
        if not mr_files:
            raise FileNotFoundError(f"No MR files found in {self.data_dir / 'mr'}. Place .nii.gz files there first.")
        self.reference_nifti_path = str(mr_files[0])  # Store for later use
        volume = load_nifti(self.reference_nifti_path)
        return volume.shape
        
    
    def process_heatmaps(self, sigma: float = 2.0):
        """Process heatmaps for both MR and synthetic US modalities."""
        logger.info("Creating heatmaps...")

        volume_shape = self.get_reference_volume_shape()
        logger.info("Reference volume shape: %s", volume_shape)

        logger.info("Processing MR modality...")
        mr_heatmap = self.create_frequency_heatmap(volume_shape, 'mr', sigma)

        logger.info("Processing synthetic US modality...")
        synthetic_us_heatmap = self.create_frequency_heatmap(volume_shape, 'synthetic_us', sigma)

        save_nifti(mr_heatmap, self.output_dir / 'mr_heatmap.nii.gz', self.reference_nifti_path)
        save_nifti(synthetic_us_heatmap, self.output_dir / 'synthetic_us_heatmap.nii.gz', self.reference_nifti_path)

        if mr_heatmap.max() > 0 and synthetic_us_heatmap.max() > 0:
            combined_heatmap = self.combine_heatmaps(mr_heatmap, synthetic_us_heatmap)
            save_nifti(combined_heatmap, self.output_dir / 'main_heatmap.nii.gz', self.reference_nifti_path)

            # Apply weighted FOV mask — Gaussian falloff from center of synthetic US FOV
            synth_us_dir = self.data_dir / 'synthetic_us'
            if synth_us_dir.exists():
                fov_mask = np.zeros(volume_shape, dtype=bool)
                for f in sorted(synth_us_dir.glob('*.nii.gz')):
                    vol = load_nifti(str(f))
                    fov_mask |= vol > 0
                logger.info("Synthetic US FOV coverage: %d / %d voxels (%.1f%%)",
                            fov_mask.sum(), fov_mask.size, 100 * fov_mask.sum() / fov_mask.size)
                weighted_mask = self.create_weighted_fov_mask(fov_mask)
                masked_heatmap = combined_heatmap * weighted_mask
                masked_heatmap = self.normalize_map(masked_heatmap)
                save_nifti(masked_heatmap, self.output_dir / 'main_heatmap.nii.gz', self.reference_nifti_path)
                logger.info("Saved weighted FOV-masked heatmap as main_heatmap.nii.gz")
        else:
            logger.warning("One or both heatmaps are empty, skipping combined heatmap")

        logger.info("All heatmaps saved to: %s", self.output_dir)

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate keypoint heatmaps from SIFT descriptors")
    parser.add_argument("--data-dir", type=str, default="data/img",
                        help="Input data directory containing mr/ and synthetic_us/")
    parser.add_argument("--output-dir", type=str, default="data/heatmap",
                        help="Output directory for heatmaps")
    parser.add_argument("--sigma", type=float, default=2.0,
                        help="Gaussian smoothing sigma")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    processor = HeatmapProcessor(args.data_dir, args.output_dir)
    processor.process_heatmaps(sigma=args.sigma)


if __name__ == "__main__":
    main()