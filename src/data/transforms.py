import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

import torch
import numpy as np
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

class Rotate3D:
    """Apply 3D rotation to a patch around its center."""
    def __init__(
        self,
        angles: Optional[Tuple[float, float, float]] = None,
        random_angles: bool = False,
        max_angle: float = 45.0
    ):
        self.angles = angles
        self.random_angles = random_angles
        self.max_angle = max_angle

    def _get_affine_matrix(self, angles: Tuple[float, float, float]) -> torch.Tensor:
        """Create 3D affine transformation matrix."""
        # Convert angles to radians
        angles = [np.deg2rad(angle) for angle in angles]

        # Create individual rotation matrices
        Rx = torch.tensor([
            [1, 0, 0],
            [0, np.cos(angles[0]), -np.sin(angles[0])],
            [0, np.sin(angles[0]), np.cos(angles[0])]
        ], dtype=torch.float32)

        Ry = torch.tensor([
            [np.cos(angles[1]), 0, np.sin(angles[1])],
            [0, 1, 0],
            [-np.sin(angles[1]), 0, np.cos(angles[1])]
        ], dtype=torch.float32)

        Rz = torch.tensor([
            [np.cos(angles[2]), -np.sin(angles[2]), 0],
            [np.sin(angles[2]), np.cos(angles[2]), 0],
            [0, 0, 1]
        ], dtype=torch.float32)

        # Combine rotations
        R = Rz @ Ry @ Rx

        # Add translation (last column) as zeros
        affine = torch.eye(4, dtype=torch.float32)
        affine[:3, :3] = R

        # Return the 3x4 matrix
        return affine[:3, :4]

    def _generate_random_angles(self) -> Tuple[float, float, float]:
        """Generate random rotation angles."""
        return tuple(
            float(np.random.uniform(-self.max_angle, self.max_angle)) 
            for _ in range(3)
        )

    def __call__(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Apply 3D rotation to the patch.
        
        Args:
            patch: Input tensor of shape [1, H, W, D]
            
        Returns:
            Rotated tensor of shape [1, H, W, D]
        """
        if patch.dim() != 4 or patch.shape[0] != 1:
            raise ValueError(f"Expected tensor of shape [1, H, W, D], got {patch.shape}")

        # Get angles
        angles = self._generate_random_angles() if self.random_angles else self.angles
        if angles is None or all(a == 0 for a in angles):
            return patch

        # Convert patch to [N, C, D, H, W] format for grid_sample
        x = patch.unsqueeze(0).permute(0, 1, 4, 2, 3)  # [1, 1, D, H, W]

        # Get affine matrix and create grid
        affine = self._get_affine_matrix(angles)
        grid = F.affine_grid(
            affine.unsqueeze(0),  # Add batch dimension
            x.size(),             # Shape [N, C, D, H, W]
            align_corners=False
        )

        # Apply transformation
        rotated = F.grid_sample(
            x,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )

        # Convert back to original format [1, H, W, D]
        rotated = rotated.squeeze(0).permute(0, 2, 3, 1)

        return rotated
    
class CenterCrop3D:
    """Apply center crop to a 3D patch."""
    def __init__(self, output_size: Tuple[int, int, int]):
        self.output_size = output_size
        
    def __call__(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Apply center crop to the patch.
        
        Args:
            patch: Input tensor of shape [1, H, W, D]
            
        Returns:
            Cropped tensor of shape [1, output_size[0], output_size[1], output_size[2]]
        """
        _, h, w, d = patch.shape
        oh, ow, od = self.output_size
        
        # Calculate crop boundaries
        h_start = (h - oh) // 2
        w_start = (w - ow) // 2
        d_start = (d - od) // 2
        
        # Apply crop
        cropped = patch[
            :,  # Channel dimension
            h_start:h_start + oh,
            w_start:w_start + ow,
            d_start:d_start + od
        ]
        
        return cropped

class ZNormalize3D:
    """Apply Z-score normalization to a 3D patch."""
    def __call__(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Apply z-score normalization.
        
        Args:
            patch: Input tensor of shape [1, H, W, D]
            
        Returns:
            Normalized tensor of same shape
        """
        mean = patch.mean()
        std = patch.std()
        return (patch - mean) / (std + 1e-8)
    

class BatchRotate3D:
    def __init__(self, max_angle: float = 45.0):
        self.max_angle = max_angle

    def _generate_random_angles(self) -> Tuple[float, float, float]:
        return tuple(
            float(np.random.uniform(-self.max_angle, self.max_angle)) 
            for _ in range(3)
        )

    def _get_affine_matrix(self, angles: Tuple[float, float, float]) -> torch.Tensor:
        angles = [np.deg2rad(angle) for angle in angles]
        
        Rx = torch.tensor([
            [1, 0, 0],
            [0, np.cos(angles[0]), -np.sin(angles[0])],
            [0, np.sin(angles[0]), np.cos(angles[0])]
        ], dtype=torch.float32)

        Ry = torch.tensor([
            [np.cos(angles[1]), 0, np.sin(angles[1])],
            [0, 1, 0],
            [-np.sin(angles[1]), 0, np.cos(angles[1])]
        ], dtype=torch.float32)

        Rz = torch.tensor([
            [np.cos(angles[2]), -np.sin(angles[2]), 0],
            [np.sin(angles[2]), np.cos(angles[2]), 0],
            [0, 0, 1]
        ], dtype=torch.float32)

        R = Rz @ Ry @ Rx
        affine = torch.eye(4, dtype=torch.float32)
        affine[:3, :3] = R
        return affine[:3, :4]

    def rotate_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Rotate all patches in batch with same random angles."""
        angles = self._generate_random_angles()
        affine = self._get_affine_matrix(angles)
        
        rotated_batch = {}
        for key in ['mr', 'synth_us', 'us']:
            x = batch[key].permute(0, 1, 4, 2, 3)  # [B, 1, D, H, W]
            
            grid = F.affine_grid(
                affine.unsqueeze(0).repeat(x.size(0), 1, 1),
                x.size(),
                align_corners=False
            )
            
            rotated = F.grid_sample(
                x, 
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            )
            
            rotated_batch[key] = rotated.permute(0, 1, 3, 4, 2)
            
        rotated_batch.update({
            'point': batch['point'],
            'style_idx': batch['style_idx']
        })
        
        return rotated_batch