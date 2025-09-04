"""
Dataset classes for 3D medical image descriptor learning.

This module provides dataset classes for training and inference with 3D medical
images (MR and ultrasound). The datasets handle patch extraction, augmentation,
and proper sampling strategies for metric learning.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
import numpy as np
from .transforms import Rotate3D, CenterCrop3D, ZNormalize3D

# Configure module logger
logger = logging.getLogger(__name__)

class BaseDescriptorDataset(Dataset):
    """
    Base class for descriptor datasets containing common functionality.
    
    This class provides core functionality for 3D medical image patch extraction,
    point sampling, and validation. It serves as the foundation for both training
    and inference datasets.
    
    Args:
        mr: MR volume tensor of shape [H, W, D]
        us: US volume tensor of shape [H, W, D]
        heatmap: Interest point heatmap for guided sampling [H, W, D]
        fov: Field of view mask [H, W, D]
        prob_dist: Flattened probability distribution for sampling
        patch_size: Size of extracted patches (height, width, depth)
        dataset_size: Number of samples per epoch
        min_distance: Minimum distance between sampled points in voxels
        max_sampling_attempts: Maximum attempts to find valid points
        transform: Optional transform composition (deprecated, use specific transforms)
    """
    
    def __init__(
        self,
        mr: torch.Tensor,
        us: torch.Tensor,
        heatmap: torch.Tensor,
        fov: torch.Tensor,
        prob_dist: torch.Tensor,
        patch_size: Tuple[int, int, int],
        dataset_size: int = 1024,
        min_distance: float = 4.0,
        max_sampling_attempts: int = 20,
    ) -> None:
        super().__init__()
        
        # Validate inputs
        self._validate_inputs(mr, us, heatmap, fov, prob_dist, patch_size)
        
        self.mr = mr
        self.us = us
        self.heatmap = heatmap
        self.fov = fov
        self.prob_dist = prob_dist
        self.patch_size = torch.tensor(patch_size, dtype=torch.long)
        self.dataset_size = dataset_size
        self.min_distance = min_distance
        self.max_sampling_attempts = max_sampling_attempts
        
        self._half_patch = self.patch_size // 2
    
    def _validate_inputs(
        self,
        mr: torch.Tensor,
        us: torch.Tensor,
        heatmap: torch.Tensor,
        fov: torch.Tensor,
        prob_dist: torch.Tensor,
        patch_size: Tuple[int, int, int]
    ) -> None:
        """Validate input tensors and parameters."""
        # Check tensor dimensions
        expected_shape = mr.shape
        for name, tensor in [("us", us), ("heatmap", heatmap), ("fov", fov)]:
            if tensor.shape != expected_shape:
                raise ValueError(f"{name} shape {tensor.shape} doesn't match MR shape {expected_shape}")
        
        # Check probability distribution
        expected_prob_size = heatmap.numel()
        if prob_dist.numel() != expected_prob_size:
            raise ValueError(f"prob_dist size {prob_dist.numel()} doesn't match heatmap size {expected_prob_size}")
        
        # Check patch size validity
        if any(p <= 0 for p in patch_size):
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        
        if any(p >= s for p, s in zip(patch_size, mr.shape)):
            raise ValueError(f"patch_size {patch_size} too large for volume shape {mr.shape}")
    
    def _is_valid_point(self, point: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                       min_coverage: float = 0.8) -> bool:
        """
        Check if a point allows valid patch extraction with sufficient FOV coverage.
        
        Args:
            point: Point coordinates [3] as (h, w, d)
            mask: Optional mask to use instead of self.fov
            min_coverage: Minimum fraction of patch that should be within the mask
            
        Returns:
            True if point allows valid patch extraction
        """
        point = point.long()
        mask = self.fov if mask is None else mask
        
        # Check bounds
        volume_shape = torch.tensor(mask.shape, dtype=torch.long)
        if torch.any(point - self._half_patch < 0) or torch.any(point + self._half_patch >= volume_shape):
            return False
        
        # Extract patch from mask and calculate coverage
        mask_patch = mask[
            point[0] - self._half_patch[0] : point[0] + self._half_patch[0],
            point[1] - self._half_patch[1] : point[1] + self._half_patch[1],
            point[2] - self._half_patch[2] : point[2] + self._half_patch[2],
        ]
        
        coverage = torch.sum(mask_patch).float() / mask_patch.numel()
        return coverage >= min_coverage
    
    def _compute_distances(self, point: torch.Tensor, existing_points: torch.Tensor) -> torch.Tensor:
        """
        Compute Euclidean distances from a point to all existing points.
        
        Args:
            point: Point coordinates [3]
            existing_points: Existing point coordinates [N, 3]
            
        Returns:
            Distances from point to each existing point [N]
        """
        if len(existing_points) == 0:
            return torch.tensor([])
        
        point_float = point.float()
        existing_points_float = existing_points.float()
        return torch.norm(existing_points_float - point_float, dim=1)
    
    def _sample_points_efficiently(self) -> torch.Tensor:
        """
        Efficiently sample points using downsampling for large volumes.
        
        This method handles large probability distributions by downsampling the
        heatmap when necessary to avoid memory issues with multinomial sampling.
        
        Returns:
            Sampled point coordinates [dataset_size, 3]
        """
        selected_points = []
        attempt = 0
        
        # Check if we need downsampling to avoid multinomial memory issues
        max_categories = 2**24  # PyTorch multinomial limit
        needs_downsampling = self.prob_dist.numel() > max_categories
        
        if needs_downsampling:
            selected_points = self._sample_with_downsampling()
        else:
            selected_points = self._sample_direct()
        
        if len(selected_points) < self.dataset_size:
            raise RuntimeError(
                f'Could only find {len(selected_points)} valid points '
                f'after {self.max_sampling_attempts} attempts. '
                f'Consider reducing min_distance or increasing max_sampling_attempts.'
            )
        
        return torch.stack(selected_points)
    
    def _sample_with_downsampling(self) -> List[torch.Tensor]:
        """Sample points using downsampled heatmap for large volumes."""
        h, w, d = self.heatmap.shape
        total_voxels = h * w * d
        
        # Calculate downsampling factor
        downsample_factor = int(np.ceil(np.power(total_voxels / (2**24), 1/3)))
        
        # Downsample heatmap using average pooling
        downsampled_heatmap = F.avg_pool3d(
            self.heatmap.unsqueeze(0).unsqueeze(0), 
            kernel_size=downsample_factor,
            stride=downsample_factor
        ).squeeze()
        
        # Create probability distribution from downsampled heatmap
        flat_downsampled = downsampled_heatmap.flatten()
        downsampled_prob_dist = flat_downsampled / (flat_downsampled.sum() + 1e-8)
        
        selected_points = []
        attempt = 0
        
        while len(selected_points) < self.dataset_size and attempt < self.max_sampling_attempts:
            num_points_needed = self.dataset_size - len(selected_points)
            
            # Sample more points than needed to account for invalid ones
            sampled_indices = torch.multinomial(
                downsampled_prob_dist,
                num_samples=min(num_points_needed * 8, len(downsampled_prob_dist)),
                replacement=True,
            )
            
            # Convert indices to coordinates in downsampled space
            downsampled_points = torch.stack(
                torch.unravel_index(sampled_indices, downsampled_heatmap.shape)
            ).T
            
            # Convert to original space coordinates
            points = downsampled_points * downsample_factor + downsample_factor // 2
            points = points[torch.randperm(points.size(0))]
            
            # Validate and add points
            selected_points.extend(self._filter_valid_points(points, selected_points))
            attempt += 1
        
        return selected_points
    
    def _sample_direct(self) -> List[torch.Tensor]:
        """Sample points directly from the original heatmap."""
        selected_points = []
        attempt = 0
        
        while len(selected_points) < self.dataset_size and attempt < self.max_sampling_attempts:
            num_points_needed = self.dataset_size - len(selected_points)
            
            # Sample indices from probability distribution
            sampled_indices = torch.multinomial(
                self.prob_dist,
                num_samples=min(num_points_needed * 8, len(self.prob_dist)),
                replacement=True,
            )
            
            # Convert to coordinates and shuffle
            points = torch.stack(torch.unravel_index(sampled_indices, self.heatmap.shape)).T
            points = points[torch.randperm(points.size(0))]
            
            # Validate and add points
            selected_points.extend(self._filter_valid_points(points, selected_points))
            attempt += 1
        
        return selected_points
    
    def _filter_valid_points(self, candidate_points: torch.Tensor, 
                           selected_points: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Filter candidate points based on validity and distance constraints.
        
        Args:
            candidate_points: Candidate point coordinates [N, 3]
            selected_points: Already selected points
            
        Returns:
            List of valid points that can be added
        """
        valid_points = []
        
        for point in candidate_points:
            if len(valid_points) + len(selected_points) >= self.dataset_size:
                break
                
            # Check if point allows valid patch extraction
            if not self._is_valid_point(point):
                continue
            
            # Check minimum distance constraint
            all_existing = selected_points + valid_points
            if len(all_existing) > 0:
                distances = self._compute_distances(point, torch.stack(all_existing))
                if torch.min(distances) < self.min_distance:
                    continue
            
            valid_points.append(point)
        
        return valid_points
    
    def _extract_patch(self, volume: torch.Tensor, center: torch.Tensor, 
                      enlarge_factor: float = 1.0) -> torch.Tensor:
        """
        Extract patch from volume centered at given point.
        
        Args:
            volume: Source volume tensor [H, W, D]
            center: Center coordinates [3]
            enlarge_factor: Factor to enlarge patch (useful for rotation)
            
        Returns:
            Extracted patch [1, H_patch, W_patch, D_patch]
        """
        # Calculate patch size (potentially enlarged for augmentation)
        if enlarge_factor > 1.0:
            half_patch = (self.patch_size.float() * enlarge_factor / 2).long()
        else:
            half_patch = self._half_patch
        
        # Convert coordinates to integers for indexing
        h, w, d = center.long()
        
        # Extract patch with bounds checking
        h_start = max(0, h - half_patch[0])
        h_end = min(volume.shape[0], h + half_patch[0])
        w_start = max(0, w - half_patch[1])
        w_end = min(volume.shape[1], w + half_patch[1])
        d_start = max(0, d - half_patch[2])
        d_end = min(volume.shape[2], d + half_patch[2])
        
        patch = volume[h_start:h_end, w_start:w_end, d_start:d_end]
        
        # Add channel dimension
        return patch.unsqueeze(0)


class DescriptorTrainingDataset(BaseDescriptorDataset):
    """
    Dataset for training descriptor models with paired MR and synthetic US images.
    
    This dataset provides pairs of MR and synthetic US patches for metric learning.
    
    Args:
        mr: MR volume tensor [H, W, D]
        synth_us: List of synthetic US volumes [H, W, D] with different styles
        heatmap: Interest point heatmap for guided sampling [H, W, D]
        fov: Field of view mask [H, W, D]
        prob_dist: Flattened probability distribution for sampling
        patch_size: Size of extracted patches (height, width, depth)
        dataset_size: Number of samples per epoch
        min_distance: Minimum distance between sampled points in voxels
        max_sampling_attempts: Maximum attempts to find valid points
        style_idx: Index of synthetic US style to use
        augment: Whether to apply rotation augmentation
        max_angle: Maximum rotation angle in degrees
        initial_angle: Initial rotation angle for curriculum learning
        angle_warmup_epochs: Number of epochs to reach max_angle
    """
    
    def __init__(
        self,
        mr: torch.Tensor,
        synth_us: List[torch.Tensor],
        heatmap: torch.Tensor,
        fov: torch.Tensor,
        prob_dist: torch.Tensor,
        patch_size: Tuple[int, int, int],
        dataset_size: int = 1024,
        min_distance: float = 4.0,
        max_sampling_attempts: int = 20,
        style_idx: int = 0,
        augment: bool = False,
        max_angle: float = 45.0,
        initial_angle: float = 5.0,
        angle_warmup_epochs: int = 200,
    ) -> None:
        # Create dummy US tensor for BaseDescriptorDataset (not used in training)
        dummy_us = torch.zeros_like(mr)
        
        super().__init__(
            mr=mr, 
            us=dummy_us,  # Pass dummy US tensor 
            heatmap=heatmap, 
            fov=fov, 
            prob_dist=prob_dist, 
            patch_size=patch_size,
            dataset_size=dataset_size, 
            min_distance=min_distance, 
            max_sampling_attempts=max_sampling_attempts,
        )
        
        # Validate synthetic US inputs
        if not synth_us:
            raise ValueError("synth_us list cannot be empty")
        if style_idx >= len(synth_us):
            raise ValueError(f"style_idx {style_idx} out of range for {len(synth_us)} styles")
        
        self.synth_us = synth_us
        self.style_idx = style_idx
        self.num_styles = len(synth_us)
        self.augment = augment
        
        # Curriculum learning parameters
        self.max_angle = max_angle
        self.initial_angle = initial_angle
        self.angle_warmup_epochs = angle_warmup_epochs
        self.current_max_angle = initial_angle
        self.current_epoch = 0
        self.current_angles: Optional[Tuple[float, float, float]] = None
        
        # Initialize transforms
        self.crop = CenterCrop3D(output_size=self.patch_size.tolist())
        self.normalize = ZNormalize3D()
        self.rotate = Rotate3D(max_angle=self.current_max_angle)
        
        # Sample initial points
        self.points = self._sample_points_efficiently()
    
    def update_angle(self, epoch: int) -> None:
        """
        Update maximum rotation angle based on current epoch for curriculum learning.
        
        Args:
            epoch: Current training epoch
        """
        self.current_epoch = epoch
        
        if epoch < self.angle_warmup_epochs:
            # Linear increase from initial_angle to max_angle
            progress = epoch / self.angle_warmup_epochs
            self.current_max_angle = self.initial_angle + (self.max_angle - self.initial_angle) * progress
        else:
            self.current_max_angle = self.max_angle
        
        # Update rotate transform with new max angle
        self.rotate.max_angle = self.current_max_angle
    
    def resample_points(self) -> None:
        """Resample points and generate new rotation angles for the next epoch."""
        self.points = self._sample_points_efficiently()
        
        if self.augment:
            self.current_angles = self.rotate._generate_random_angles()
        else:
            self.current_angles = None
    
    def _process_mr_patch(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Process MR patch with optional rotation, cropping, and normalization.
        
        Args:
            patch: Raw MR patch [1, H, W, D]
            
        Returns:
            Processed patch [1, patch_size[0], patch_size[1], patch_size[2]]
        """
        if self.augment and self.current_angles is not None:
            # Apply rotation using precomputed angles
            affine = self.rotate._get_affine_matrix(self.current_angles)
            
            # Reshape for grid_sample: [1, 1, D, H, W]
            x = patch.unsqueeze(0).permute(0, 1, 4, 2, 3)
            
            # Generate sampling grid and apply transformation
            grid = F.affine_grid(
                affine.unsqueeze(0),
                x.size(),
                align_corners=False
            )
            
            # Apply transformation
            patch = F.grid_sample(
                x,
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            ).squeeze(0).permute(0, 2, 3, 1)  # Back to [1, H, W, D]
        
        # Apply cropping and normalization
        patch = self.crop(patch)
        patch = self.normalize(patch)
        return patch
    
    def _process_us_patch(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Process US patch with cropping and normalization (no rotation).
        
        Args:
            patch: Raw US patch [1, H, W, D]
            
        Returns:
            Processed patch [1, patch_size[0], patch_size[1], patch_size[2]]
        """
        patch = self.crop(patch)
        patch = self.normalize(patch)
        return patch
    
    def __len__(self) -> int:
        """Return the number of points in the dataset."""
        return len(self.points)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample consisting of MR and synthetic US patches.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - mr: Processed MR patch (with rotation if augmented)
                - synth_us: Processed synthetic US patch
                - point: Point coordinates [3]
                - style_idx: Style index used
        """
        point = self.points[idx]
        
        # Determine patch extraction size (larger if augmenting)
        enlarge_factor = 1.5 if self.augment else 1.0
        
        # Extract patches
        mr_patch = self._extract_patch(self.mr, point, enlarge_factor)
        synth_patch = self._extract_patch(self.synth_us[self.style_idx], point)
        
        # Process patches
        mr_patch = self._process_mr_patch(mr_patch)
        synth_patch = self._process_us_patch(synth_patch)
        
        return {
            'mr': mr_patch,
            'synth_us': synth_patch,
            'point': point,
            'style_idx': torch.tensor(self.style_idx, dtype=torch.long)
        }

class DescriptorInferenceDataset(BaseDescriptorDataset):
    """
    Dataset for descriptor inference on both MR and real US volumes.
    
    Args:
        mr: MR volume tensor [H, W, D]
        us: US volume tensor [H, W, D]
        heatmap: Probability heatmap for sampling [H, W, D]
        fov: Field-of-view mask [H, W, D]
        prob_dist: Probability distribution for sampling [H, W, D]
        patch_size: Size of patches to extract [H, W, D]
        dataset_size: Number of MR points to sample (default: 1024)
        min_distance: Minimum distance between sampled points (default: 4.0)
        max_sampling_attempts: Maximum attempts for point sampling (default: 20)
        mr_points: Pre-defined MR points (optional)
        us_points: Pre-defined US points (optional)
        grid_spacing: Spacing for US grid generation (default: 8)
        
    Example:
        >>> dataset = DescriptorInferenceDataset(
        ...     mr=mr_volume,
        ...     us=us_volume,
        ...     heatmap=prob_map,
        ...     fov=fov_mask,
        ...     prob_dist=prob_dist,
        ...     patch_size=(64, 64, 64),
        ...     grid_spacing=8
        ... )
        >>> sample = dataset[0]
        >>> patch, point, modality = sample['patch'], sample['point'], sample['modality']
    """
    
    def __init__(
        self,
        mr: torch.Tensor,
        us: torch.Tensor,
        heatmap: torch.Tensor,
        fov: torch.Tensor,
        prob_dist: torch.Tensor,
        patch_size: Tuple[int, int, int],
        dataset_size: int = 1024,
        min_distance: float = 4.0,
        max_sampling_attempts: int = 20,
        mr_points: Optional[torch.Tensor] = None,
        us_points: Optional[torch.Tensor] = None,
        grid_spacing: int = 8,
    ) -> None:
        """Initialize inference dataset with both MR and US sampling."""
        super().__init__(
            mr=mr,
            us=us,
            heatmap=heatmap,
            fov=fov,
            prob_dist=prob_dist,
            patch_size=patch_size,
            dataset_size=dataset_size,
            min_distance=min_distance,
            max_sampling_attempts=max_sampling_attempts,
        )
        
        self.grid_spacing = grid_spacing
        self.normalize = ZNormalize3D()
        
        # Set MR points or sample new ones
        self.mr_points = mr_points if mr_points is not None else self._sample_points_efficiently()
        
        # Generate or set US points
        self.us_points = us_points if us_points is not None else self._generate_us_grid()
        
        # Create combined point list with modality tracking
        self.modality = ['mr'] * len(self.mr_points) + ['us'] * len(self.us_points)
        self.points = torch.cat([self.mr_points, self.us_points], dim=0)
        
        logger.info(
            f"Initialized inference dataset: {len(self.mr_points)} MR points, "
            f"{len(self.us_points)} US points."
        )
    
    def _generate_us_grid(self) -> torch.Tensor:
        """
        Generate dense grid points within US volume for comprehensive coverage.
        
        Creates a regular grid covering valid US regions, ensuring all patches
        can be extracted without boundary issues.
        
        Returns:
            Tensor of valid grid points [N, 3] in (h, w, d) coordinates
        """
        h, w, d = self.us.shape
        us_mask = self.us > 0
        
        # Calculate valid coordinate ranges
        h_coords = torch.arange(
            self._half_patch[0], 
            h - self._half_patch[0], 
            self.grid_spacing,
            dtype=torch.float32
        )
        w_coords = torch.arange(
            self._half_patch[1], 
            w - self._half_patch[1], 
            self.grid_spacing,
            dtype=torch.float32
        )
        d_coords = torch.arange(
            self._half_patch[2], 
            d - self._half_patch[2], 
            self.grid_spacing,
            dtype=torch.float32
        )
        
        # Create meshgrid and flatten
        grid_h, grid_w, grid_d = torch.meshgrid(h_coords, w_coords, d_coords, indexing='ij')
        candidate_points = torch.stack([grid_h, grid_w, grid_d], dim=-1).reshape(-1, 3)
        
        # Filter points based on US volume coverage
        valid_points = []
        for point in candidate_points:
            h_idx, w_idx, d_idx = point.long()
            
            # Check if patch region has sufficient US data
            h_start = h_idx - self._half_patch[0]
            h_end = h_idx + self._half_patch[0]
            w_start = w_idx - self._half_patch[1]
            w_end = w_idx + self._half_patch[1]
            d_start = d_idx - self._half_patch[2]
            d_end = d_idx + self._half_patch[2]
            
            patch_region = us_mask[h_start:h_end, w_start:w_end, d_start:d_end]
            coverage = patch_region.float().mean()
            
            # Include points with sufficient US coverage
            if coverage > 0.1:  # At least 10% US data
                valid_points.append(point)
        
        if not valid_points:
            logger.warning("No valid US grid points found, using center point")
            center = torch.tensor([h // 2, w // 2, d // 2], dtype=torch.float32)
            return center.unsqueeze(0)
        
        valid_points = torch.stack(valid_points)
        logger.info(f"Generated {len(valid_points)} valid US grid points")
        
        return valid_points
    
    def __len__(self) -> int:
        """Return the total number of points (MR + US)."""
        return len(self.points)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the point to retrieve
            
        Returns:
            Dictionary containing:
                - patch: Processed patch tensor [1, H, W, D]
                - point: Point coordinates [3]
                - modality: Modality type ('mr' or 'us')
        """
        point = self.points[idx]
        modality = self.modality[idx]
        
        # Select appropriate volume
        volume = self.mr if modality == 'mr' else self.us
        
        # Extract raw patch
        patch = self._extract_patch(volume, point)
        patch = self.normalize(patch)

        return {
            'patch': patch,
            'point': point,
            'modality': modality
        }