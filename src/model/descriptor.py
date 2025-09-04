"""
Lightning module for 3D medical image descriptor learning.

This module implements a PyTorch Lightning model for learning descriptors from
3D medical images (MR and ultrasound) using metric learning approaches.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import numpy as np
import torch
import pytorch_lightning as pl
from .networks import resnet18_3d_encoder
from .losses import InfoNCELoss, CurriculumTripletLoss
from .matcher import KNNMatcher

class Descriptor(pl.LightningModule):
    """
    Lightning module for 3D descriptor learning.
    
    This class implements a PyTorch Lightning model for learning descriptors from
    3D medical images using metric learning.
    
    Args:
        # Model parameters
        backbone: Network backbone type ('resnet18', 'resnet34', etc.)
        out_dim: Output descriptor dimension
        input_channels: Number of input channels
        pretrained_path: Path to pretrained weights (optional)
        freeze_backbone: Whether to freeze backbone weights

        # Loss parameters
        loss_type: Type of loss function ('triplet', 'infonce')
        margin: Margin for triplet loss
        temperature: Temperature for InfoNCE loss
        warmup_epochs: Number of epochs for curriculum learning
        spatial_weight: Weight for spatial information in triplet loss
        
        # Optimizer parameters
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        max_epochs: Maximum number of training epochs
        eta_min: Minimum learning rate for cosine annealing
        
        # Evaluation parameters  
        knn_k: Number of nearest neighbors for evaluation
        distance_threshold: Distance threshold for matching
        ratio_threshold: Ratio threshold for matching
        mutual: Whether to use mutual nearest neighbors
        metric: Distance metric for matching
        max_distance: Maximum distance for correct matches
        
    Note:
        This model does not use validation during training since real US data
        is not available during training (only synthetic US). Evaluation is
        performed during testing with real US data.
    """
    
    def __init__(
        self,
        # Model parameters
        out_dim: int = 512,
        input_channels: int = 1,
        
        # Loss parameters
        loss_type: str = 'triplet',
        margin: float = 1.0,
        temperature: float = 0.1,
        warmup_epochs: int = 200,
        spatial_weight: float = 0.3,
        max_spatial_dist: float = 48.0,

        # Optimizer parameters
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        max_epochs: int = 1000,
        eta_min: float = 1e-6,
        
        # Evaluation parameters
        knn_k: int = 1,
        distance_threshold: float = float('inf'),
        ratio_threshold: float = 0.8,
        mutual: bool = True,
        metric: str = 'euclidean',
        max_distance: float = 5.0,
        
        **kwargs
    ) -> None:
        super().__init__()
        
        # Save all hyperparameters
        self.save_hyperparameters()
        
        # Initialize descriptor network
        self.model = resnet18_3d_encoder(
            input_channels=input_channels,
            feature_dim=out_dim
        )
        
        # Initialize loss function
        if loss_type == 'triplet':
            self.loss_fn = CurriculumTripletLoss(
                margin=margin,
                warmup_epochs=warmup_epochs,
                spatial_weight=spatial_weight,
                max_spatial_dist=max_spatial_dist
            )
        elif loss_type == 'infonce':
            self.loss_fn = InfoNCELoss(temperature=temperature)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Initialize matcher
        self.matcher = KNNMatcher(
            k=knn_k,
            distance_threshold=distance_threshold,
            ratio_threshold=ratio_threshold, 
            mutual=mutual,
            metric=metric,
            evaluation_threshold=max_distance
        )
            
        # Store evaluation parameters
        self.max_distance = max_distance
        
        # Initialize test outputs storage
        self.test_step_outputs: List[Dict[str, Any]] = []
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizers and learning rate schedulers.
        
        Returns:
            Dictionary containing optimizer and scheduler configuration
        """
        # Define optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        # Define scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=self.hparams.eta_min
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            }
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape [B, C, D, H, W]
            
        Returns:
            Feature descriptors of shape [B, out_dim]
        """
        return self.model(x)
    
    def on_train_epoch_start(self) -> None:
        """
        Called at the start of each training epoch.
        
        Updates dataset parameters like rotation angles and resamples points.
        """
        dataset = self.trainer.train_dataloader.dataset
        
        # Update maximum rotation angle
        dataset.update_angle(self.current_epoch)
        
        # Choose random style for this epoch
        new_style = torch.randint(0, dataset.num_styles, (1,)).item()
        dataset.style_idx = new_style
        
        # Resample points
        dataset.resample_points()

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step for one batch using MR and synthetic US data pairs.
        
        Args:
            batch: Dictionary containing 'mr' and 'synth_us' tensors
            batch_idx: Index of the current batch
            
        Returns:
            Loss tensor for this batch
        """
        # Forward pass
        anchor_output = self(batch['mr'])         
        positive_output = self(batch['synth_us'])
        
        # Update loss function with current epoch for curriculum learning
        self.loss_fn.update_epoch(self.current_epoch)
        
        # Calculate loss
        loss, components = self.loss_fn(anchor_output, positive_output)
        
        # Log metrics
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        if isinstance(components, dict):
            train_components = {f'train/{k}': v for k, v in components.items()}
            self.log_dict(train_components, on_step=False, on_epoch=True)
        
        return loss

    # Note: No validation_step() is implemented because training uses synthetic US data
    # while validation would require real US data, which is only available during testing.
    # This is a design choice to separate training (MR + synthetic US) from evaluation (MR + real US).

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """
        Process a single test batch.
        
        Args:
            batch: Dictionary containing test data
            batch_idx: Index of the current batch
        """
        # Extract descriptors
        descriptors = self(batch['patch'])
        
        # Store descriptors along with metadata
        self.test_step_outputs.append({
            'descriptors': descriptors.cpu(),
            'points': batch['point'].cpu(),
            'modalities': batch['modality'],  # List of strings: 'mr' or 'us'
            'patches': batch['patch'].cpu()  # Save patches for visualization
        })
    
    def on_test_epoch_start(self) -> None:
        """Initialize storage for test outputs."""
        self.test_step_outputs = []
    
    def on_test_epoch_end(self) -> None:
        """
        Process all test outputs and compute matches.
        
        Concatenates all test outputs, performs matching, and evaluates results.
        """
        if not self.test_step_outputs:
            return
            
        # Concatenate all outputs
        all_descriptors = torch.cat([x['descriptors'] for x in self.test_step_outputs])
        all_points = torch.cat([x['points'] for x in self.test_step_outputs])
        all_patches = torch.cat([x['patches'] for x in self.test_step_outputs])
        all_modalities = sum([x['modalities'] for x in self.test_step_outputs], [])
        
        # Split into MR and US
        mr_mask = torch.tensor([m == 'mr' for m in all_modalities])
        us_mask = ~mr_mask
        
        mr_descriptors = all_descriptors[mr_mask]
        us_descriptors = all_descriptors[us_mask]
        mr_points = all_points[mr_mask]
        us_points = all_points[us_mask]
        mr_patches = all_patches[mr_mask]
        us_patches = all_patches[us_mask]
        
        if len(mr_descriptors) == 0 or len(us_descriptors) == 0:
            print("No MR or US descriptors found for matching")
            return
        
        # Perform matching and evaluation
        try:
            match_pairs, metrics = self.matcher.match_and_evaluate(
                src_descriptors=mr_descriptors,
                tgt_descriptors=us_descriptors,
                src_points=mr_points,
                tgt_points=us_points
            )
            
            # Log test metrics
            test_metrics = {f'test/{k}': v for k, v in metrics.items()}
            self.log_dict(test_metrics, on_step=False, on_epoch=True)
            
        except Exception as e:
            print(f"Error in test matching: {e}")