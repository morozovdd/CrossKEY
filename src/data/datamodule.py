"""
PyTorch Lightning DataModule for 3D medical image descriptor learning.

This module provides a Lightning DataModule for organizing training and inference
data for 3D medical image descriptor learning with MR and ultrasound volumes.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import nibabel as nib
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .dataset import DescriptorTrainingDataset, DescriptorInferenceDataset
from src.utils.utils import load_nifti

# Configure module logger
logger = logging.getLogger(__name__)


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """
    Normalize volume using min-max normalization.
    
    Args:
        volume: Input volume array
        
    Returns:
        Normalized volume with values in range [0, 1]
    """
    volume_min = volume.min()
    volume_max = volume.max()
    
    if volume_max == volume_min:
        logger.warning("Volume has constant values, returning zeros")
        return np.zeros_like(volume)
    
    return (volume - volume_min) / (volume_max - volume_min)

class DescriptorDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for 3D medical image descriptor learning.
    
    This DataModule efficiently handles loading and preprocessing of data for both
    training and inference. It only loads the data required for each stage:
    - Training: MR + Synthetic US + Heatmap
    - Inference: MR + Real US + Heatmap
    
    Args:
        data_dir: Root directory containing the data with the following structure:
            ├── img/
            │   ├── mr/*.nii.gz
            │   └── us/*.nii.gz
            ├── img/synthetic_us/*.nii.gz
            └── heatmap/*.npy
            
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        patch_size: Size of patches to extract [H, W, D] (default: [32, 32, 32])
        num_samples: Number of training samples per epoch (default: 1024)
        grid_spacing: Spacing for inference grid generation (default: 8)
        augment: Whether to apply data augmentation (default: False)
        max_angle: Maximum rotation angle for augmentation (default: 45.0)
        initial_angle: Initial rotation angle for curriculum learning (default: 5.0)
        angle_warmup_epochs: Epochs for angle warmup (default: 200)
        mr_points: Pre-defined MR sample points for inference (optional)
        us_points: Pre-defined US sample points for inference (optional)
        mr_path: Custom MR file path (optional)
        us_path: Custom US file path (optional, only needed for inference)
        synth_us_path: Custom synthetic US directory path (optional)
        heatmap_path: Custom heatmap file path (optional)
        
    Example:
        >>> # For training (only needs MR + synthetic US)
        >>> datamodule = DescriptorDataModule(
        ...     data_dir='/path/to/data',
        ...     batch_size=64,
        ...     num_workers=4,
        ...     patch_size=[32, 32, 32],
        ...     augment=True
        ... )
        >>> datamodule.setup('fit')
        >>> train_loader = datamodule.train_dataloader()
        
        >>> # For inference (needs MR + real US)
        >>> datamodule.setup('test')
        >>> test_loader = datamodule.test_dataloader()
    """
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        patch_size: Tuple[int, int, int] = (32, 32, 32),
        num_samples: int = 1024,
        grid_spacing: int = 8,
        augment: bool = False,
        max_angle: float = 45.0,
        initial_angle: float = 5.0,
        angle_warmup_epochs: int = 200,
        mr_points: Optional[torch.Tensor] = None,
        us_points: Optional[torch.Tensor] = None,
        mr_path: Optional[str] = None,
        us_path: Optional[str] = None,
        synth_us_path: Optional[str] = None,
        heatmap_path: Optional[str] = None,
    ) -> None:
        """Initialize the DataModule with configuration parameters."""
        super().__init__()
        
        # Core parameters
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.grid_spacing = grid_spacing
        
        # Data paths (optional overrides)
        self.mr_path = mr_path
        self.us_path = us_path
        self.synth_us_path = synth_us_path
        self.heatmap_path = heatmap_path
        
        # Inference-specific configuration
        self.mr_points = mr_points
        self.us_points = us_points
        
        # Training augmentation parameters
        self.augment = augment
        self.max_angle = max_angle
        self.initial_angle = initial_angle
        self.angle_warmup_epochs = angle_warmup_epochs
        
        # Dataset placeholders
        self.train_dataset: Optional[DescriptorTrainingDataset] = None
        self.test_dataset: Optional[DescriptorInferenceDataset] = None
        
        # Loaded data cache
        self._mr_volume: Optional[torch.Tensor] = None
        self._us_volume: Optional[torch.Tensor] = None
        self._synth_us_volumes: Optional[List[torch.Tensor]] = None
        self._heatmap: Optional[torch.Tensor] = None
        self._fov_mask: Optional[torch.Tensor] = None
        self._prob_dist: Optional[torch.Tensor] = None
        
        logger.info(
            f"Initialized DataModule: batch_size={batch_size}, "
            f"patch_size={patch_size}, augment={augment}"
        )
    
    def _pad_volume(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Pad volume with zeros based on patch size to ensure valid patch extraction.
        
        Args:
            volume: Input volume tensor [H, W, D]
            
        Returns:
            Padded volume tensor with additional padding on all sides
        """
        half_patch = [p // 2 for p in self.patch_size]
        padding = (
            half_patch[2],  # Pad left (D)
            half_patch[2],  # Pad right (D)
            half_patch[1],  # Pad top (W)
            half_patch[1],  # Pad bottom (W)
            half_patch[0],  # Pad front (H)
            half_patch[0],  # Pad back (H)
        )
        return F.pad(volume, padding, mode='constant', value=0)

    def _unpad_volume(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Remove padding from volume to restore original size.
        
        Args:
            volume: Padded volume tensor [H, W, D]
            
        Returns:
            Unpadded volume tensor
        """
        half_patch = [p // 2 for p in self.patch_size]
        return volume[
            half_patch[0]:-half_patch[0],
            half_patch[1]:-half_patch[1], 
            half_patch[2]:-half_patch[2]
        ]
    
    def _load_and_normalize_volume(self, file_path: Union[str, Path]) -> torch.Tensor:
        """
        Load and normalize a volume from NIfTI file.
        
        Args:
            file_path: Path to NIfTI file
            
        Returns:
            Normalized volume as torch tensor
        """
        volume_data = load_nifti(file_path)
        normalized_data = normalize_volume(volume_data)
        return torch.from_numpy(normalized_data).float()
    
    def _find_mr_file(self) -> Path:
        """Find MR file in the data directory."""
        if self.mr_path is not None:
            mr_file = Path(self.mr_path)
            if not mr_file.exists():
                raise FileNotFoundError(f"Specified MR file not found: {mr_file}")
            return mr_file
            
        mr_candidates = list(self.data_dir.glob('img/mr/*.nii.gz'))
        if not mr_candidates:
            raise FileNotFoundError(f"No MR file found in {self.data_dir}/img/mr/")
        
        if len(mr_candidates) > 1:
            logger.warning(f"Multiple MR files found, using: {mr_candidates[0].name}")
        
        return mr_candidates[0]
    
    def _find_us_file(self) -> Path:
        """Find real US file in the data directory (only needed for inference)."""
        if self.us_path is not None:
            us_file = Path(self.us_path)
            if not us_file.exists():
                raise FileNotFoundError(f"Specified US file not found: {us_file}")
            return us_file
            
        us_candidates = list(self.data_dir.glob('img/us/*.nii.gz'))
        if not us_candidates:
            raise FileNotFoundError(f"No US file found in {self.data_dir}/img/us/")
        
        if len(us_candidates) > 1:
            logger.warning(f"Multiple US files found, using: {us_candidates[0].name}")
            
        return us_candidates[0]
    
    def _find_synthetic_us_files(self) -> List[Path]:
        """Find synthetic US files (only needed for training)."""
        if self.synth_us_path is not None:
            synth_us_dir = Path(self.synth_us_path)
        else:
            synth_us_dir = self.data_dir / 'img' / 'synthetic_us'
        
        if not synth_us_dir.exists():
            raise FileNotFoundError(f"Synthetic US directory not found: {synth_us_dir}")
        
        synth_us_files = list(synth_us_dir.glob('*.nii.gz'))
        if not synth_us_files:
            raise FileNotFoundError(f"No synthetic US files found in: {synth_us_dir}")
        
        # Sort files for consistent ordering
        synth_us_files.sort()
        logger.info(f"Found {len(synth_us_files)} synthetic US files")
        
        return synth_us_files
    
    def _find_heatmap_file(self) -> Path:
        """Find heatmap file in the data directory."""
        if self.heatmap_path is not None:
            heatmap_file = Path(self.heatmap_path)
            if not heatmap_file.exists():
                raise FileNotFoundError(f"Specified heatmap file not found: {heatmap_file}")
            return heatmap_file
        
        heatmap_candidates = list(self.data_dir.glob('heatmap/main_heatmap.nii.gz'))
        
        if not heatmap_candidates:
            raise FileNotFoundError(f"No heatmap file found in {self.data_dir}/heatmap/")
        
        if len(heatmap_candidates) > 1:
            logger.warning(f"Multiple heatmap files found, using: {heatmap_candidates[0].name}")
            
        return heatmap_candidates[0]
    
    def _load_heatmap_data(self, heatmap_file: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load heatmap and create FOV mask and probability distribution."""
        heatmap_data = load_nifti(heatmap_file)
        heatmap = torch.from_numpy(heatmap_data).float()
        
        # Create FOV mask (binary mask where heatmap > 0)
        fov_mask = (heatmap > 0).float()
        
        # Create probability distribution from heatmap
        flat_heatmap = heatmap.flatten()
        prob_dist = flat_heatmap / flat_heatmap.sum()
        
        return heatmap, fov_mask, prob_dist
    
    def _load_mr_volume(self) -> torch.Tensor:
        """Load and cache MR volume."""
        if self._mr_volume is None:
            mr_file = self._find_mr_file()
            logger.info(f"Loading MR volume: {mr_file.name}")
            self._mr_volume = self._load_and_normalize_volume(mr_file)
            self._mr_volume = self._pad_volume(self._mr_volume)
            logger.info(f"MR volume shape: {self._mr_volume.shape}")
        return self._mr_volume
    
    def _load_us_volume(self) -> torch.Tensor:
        """Load and cache real US volume (only for inference)."""
        if self._us_volume is None:
            us_file = self._find_us_file()
            logger.info(f"Loading US volume: {us_file.name}")
            self._us_volume = self._load_and_normalize_volume(us_file)
            self._us_volume = self._pad_volume(self._us_volume)
            logger.info(f"US volume shape: {self._us_volume.shape}")
        return self._us_volume
    
    def _load_synthetic_us_volumes(self) -> List[torch.Tensor]:
        """Load and cache synthetic US volumes (only for training)."""
        if self._synth_us_volumes is None:
            synth_us_files = self._find_synthetic_us_files()
            logger.info(f"Loading {len(synth_us_files)} synthetic US volumes")
            
            self._synth_us_volumes = []
            for file in synth_us_files:
                volume = self._load_and_normalize_volume(file)
                volume = self._pad_volume(volume)
                self._synth_us_volumes.append(volume)
                
            logger.info(f"Synthetic US volumes loaded, shapes: {[v.shape for v in self._synth_us_volumes[:3]]}...")
        return self._synth_us_volumes
    
    def _load_heatmap_data_cached(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load and cache heatmap data."""
        if self._heatmap is None:
            heatmap_file = self._find_heatmap_file()
            logger.info(f"Loading heatmap: {heatmap_file.name}")
            
            self._heatmap, self._fov_mask, _ = self._load_heatmap_data(heatmap_file)
            self._heatmap = self._pad_volume(self._heatmap)
            self._fov_mask = self._pad_volume(self._fov_mask)
            
            # Create probability distribution from padded heatmap
            flat_heatmap = self._heatmap.flatten()
            self._prob_dist = flat_heatmap / flat_heatmap.sum()
            
            logger.info(f"Heatmap shape: {self._heatmap.shape}")
            
        return self._heatmap, self._fov_mask, self._prob_dist
        
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up datasets for training or testing.
        
        Args:
            stage: Stage identifier ('fit' for training, 'test' for testing)
        """
        logger.info(f"Setting up DataModule for stage: {stage}")
        
        if stage == 'fit':
            # Training stage: Load MR + Synthetic US + Heatmap
            mr = self._load_mr_volume()
            synth_us = self._load_synthetic_us_volumes()
            heatmap, fov_mask, prob_dist = self._load_heatmap_data_cached()
            
            logger.info("Creating training dataset (MR + Synthetic US)")
            self.train_dataset = DescriptorTrainingDataset(
                mr=mr,
                synth_us=synth_us,
                heatmap=heatmap,
                fov=fov_mask,
                prob_dist=prob_dist,
                patch_size=self.patch_size,
                dataset_size=self.num_samples,
                augment=self.augment,
                max_angle=self.max_angle,
                initial_angle=self.initial_angle,
                angle_warmup_epochs=self.angle_warmup_epochs
            )
            logger.info(f"Created training dataset with {len(self.train_dataset)} samples")
        
        elif stage == 'test':
            # Inference stage: Load MR + Real US + Heatmap
            mr = self._load_mr_volume()
            us = self._load_us_volume()
            heatmap, fov_mask, prob_dist = self._load_heatmap_data_cached()
            
            logger.info("Creating inference dataset (MR + Real US)")
            self.test_dataset = DescriptorInferenceDataset(
                mr=mr,
                us=us,
                heatmap=heatmap,
                fov=fov_mask,
                prob_dist=prob_dist,
                patch_size=self.patch_size,
                grid_spacing=self.grid_spacing,
                mr_points=self.mr_points,
                us_points=self.us_points,
            )
            logger.info(f"Created test dataset with {len(self.test_dataset)} samples")
        
        else:
            raise ValueError(f"Unsupported stage: {stage}. Use 'fit' or 'test'.")
    
    def train_dataloader(self) -> DataLoader:
        """
        Create training data loader.
        
        Returns:
            DataLoader for training dataset
        """
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not initialized. Call setup('fit') first.")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def test_dataloader(self) -> DataLoader:
        """
        Create test data loader.
        
        Returns:
            DataLoader for test dataset
        """
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not initialized. Call setup('test') first.")
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # No shuffle for testing
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    def get_volume_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """
        Get the shapes of loaded volumes (after padding).
        
        Returns:
            Dictionary with volume shapes for the current stage
        """
        shapes = {}
        
        if self._mr_volume is not None:
            shapes['mr'] = self._mr_volume.shape
            
        if self._us_volume is not None:
            shapes['us'] = self._us_volume.shape
            
        if self._synth_us_volumes is not None:
            shapes['synth_us'] = [v.shape for v in self._synth_us_volumes]
            
        if self._heatmap is not None:
            shapes['heatmap'] = self._heatmap.shape
            
        return shapes
    
    def clear_cache(self) -> None:
        """Clear cached volumes to free memory."""
        self._mr_volume = None
        self._us_volume = None
        self._synth_us_volumes = None
        self._heatmap = None
        self._fov_mask = None
        self._prob_dist = None
        logger.info("Cleared cached volumes")
    
    def update_training_angle(self, epoch: int) -> None:
        """
        Update rotation angle for curriculum learning.
        
        Args:
            epoch: Current training epoch
        """
        if hasattr(self, 'train_dataset') and self.train_dataset is not None:
            self.train_dataset.update_angle(epoch)
            logger.info(f"Updated rotation angle for epoch {epoch}")
    
    def resample_training_points(self) -> None:
        """Resample training points for the next epoch."""
        if hasattr(self, 'train_dataset') and self.train_dataset is not None:
            self.train_dataset.resample_points()
            logger.info("Resampled training points")