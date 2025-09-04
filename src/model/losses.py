"""
Loss functions for metric learning and contrastive learning.

This module provides various loss functions designed for metric learning tasks,
including triplet loss with curriculum learning, InfoNCE loss for contrastive
learning, and binary cross-entropy loss for similarity learning.

Classes:
    BaseLoss: Abstract base class for all loss functions
    CurriculumTripletLoss: Triplet loss with curriculum learning and spatial awareness
    InfoNCELoss: InfoNCE loss for contrastive learning
    BCELoss: Binary cross-entropy loss for similarity learning
"""

from typing import Dict, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseLoss(nn.Module):
    """
    Abstract base class for all loss functions.
    
    This class provides a consistent interface for all loss functions used in
    the project. It includes methods for curriculum learning support and
    ensures all loss functions return both loss values and component metrics.
    
    Attributes:
        current_epoch (int): Current training epoch for curriculum learning
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.current_epoch = 0
    
    def update_epoch(self, epoch: int) -> None:
        """
        Update current epoch for curriculum learning.
        
        Args:
            epoch (int): Current training epoch
        """
        self.current_epoch = epoch
    
    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute the loss and return metrics.
        
        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: Loss tensor and dictionary of metrics
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement forward method")

class CurriculumTripletLoss(BaseLoss):
    """
    Triplet loss with curriculum learning and spatial awareness.
    
    This loss function implements a triplet loss that gradually increases mining
    difficulty during training. It can optionally incorporate spatial information
    to improve negative sample selection. The curriculum starts with easier
    negatives and progressively selects harder ones as training progresses.
    
    Args:
        margin (float, optional): Margin for triplet loss. Defaults to 1.0.
        warmup_epochs (int, optional): Number of epochs for curriculum warmup. 
            Defaults to 200.
        spatial_weight (float, optional): Weight for spatial information in 
            negative selection. Defaults to 0.3.
        max_spatial_dist (float, optional): Maximum spatial distance to consider
            for normalization. Defaults to 32.0.
    
    Attributes:
        margin (float): Triplet loss margin
        warmup_epochs (int): Number of warmup epochs
        spatial_weight (float): Spatial weighting factor
        max_spatial_dist (float): Maximum spatial distance
        current_epoch (int): Current training epoch
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        warmup_epochs: int = 200,
        spatial_weight: float = 0.5,
        max_spatial_dist: float = 48.0
    ) -> None:
        super().__init__()
        self.margin = margin
        self.warmup_epochs = warmup_epochs
        self.spatial_weight = spatial_weight
        self.max_spatial_dist = max_spatial_dist
    
    def _get_mining_difficulty(self) -> float:
        """
        Get current mining difficulty based on epoch.
        
        The difficulty starts at 0 and linearly increases to 1.0 over the
        warmup period. After warmup, difficulty remains at 1.0.
        
        Returns:
            float: Mining difficulty in range [0, 1]
        """
        if self.current_epoch >= self.warmup_epochs:
            return 1.0
        return self.current_epoch / self.warmup_epochs
    
    def _compute_spatial_weights(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Compute weights based on spatial distances between points.
        
        Calculates pairwise spatial distances and converts them to weights
        for negative sample selection. Points that are spatially farther
        apart are preferred as negatives.
        
        Args:
            coordinates (torch.Tensor): Spatial coordinates of shape (N, D)
                where N is batch size and D is coordinate dimension
                
        Returns:
            torch.Tensor: Weight matrix of shape (N, N) with spatial weights
        """
        # Compute pairwise distances between points
        spatial_dist = torch.cdist(coordinates.float(), coordinates.float(), p=2)
        
        # Normalize distances to [0, 1] range
        spatial_dist = torch.clamp(spatial_dist / self.max_spatial_dist, 0, 1)
        
        # Convert to weights: points further apart should be preferred as negatives
        weights = spatial_dist / spatial_dist.max()
        
        # Mask out self-pairs
        mask = torch.eye(len(coordinates), device=coordinates.device)
        weights = weights.masked_fill(mask.bool(), float('-inf'))
        
        return weights
    
    def forward(
        self, 
        anchor: torch.Tensor, 
        positive: torch.Tensor, 
        coordinates: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Union[float, int]]]:
        """
        Compute curriculum triplet loss.
        
        Args:
            anchor (torch.Tensor): Anchor embeddings of shape (N, D)
            positive (torch.Tensor): Positive embeddings of shape (N, D)
            coordinates (torch.Tensor, optional): Spatial coordinates of shape (N, C).
                If provided, spatial information is used for negative selection.
                
        Returns:
            Tuple[torch.Tensor, Dict[str, Union[float, int]]]: 
                - Loss tensor (scalar)
                - Dictionary containing loss components and metrics:
                    - mean_pos_dist: Average positive pair distance
                    - mean_neg_dist: Average negative pair distance
                    - num_active_triplets: Number of active (non-zero) triplets
                    - triplet_ratio: Ratio of active triplets to batch size
                    - mining_difficulty: Current mining difficulty [0, 1]
        """
        batch_size = anchor.size(0)
        
        # Normalize embeddings for consistent distance computation
        anchor_norm = F.normalize(anchor, p=2, dim=1)
        positive_norm = F.normalize(positive, p=2, dim=1)
        
        # Compute positive distances (L2 squared)
        pos_dist = torch.sum((anchor_norm - positive_norm) ** 2, dim=1)
        
        # Compute all pairwise distances for potential negatives
        neg_dist_matrix = torch.cdist(anchor_norm, positive_norm, p=2)
        
        # Mask out positive pairs (diagonal elements)
        mask = torch.eye(batch_size, device=anchor.device)
        neg_dist_matrix = neg_dist_matrix.masked_fill(mask.bool(), float('inf'))
        
        # Get current mining difficulty
        difficulty = self._get_mining_difficulty()
        
        if coordinates is not None:
            # Incorporate spatial information for better negative selection
            spatial_weights = self._compute_spatial_weights(coordinates)
            
            # Combine embedding distances with spatial weights
            # Early training: prefer spatially distant negatives
            # Later training: prefer hard negatives based on embedding distance
            selection_matrix = (1 - difficulty) * spatial_weights + difficulty * (-neg_dist_matrix)
        else:
            # Without coordinates, just use embedding distances
            selection_matrix = -neg_dist_matrix
        
        # Select negatives based on combined scores
        _, neg_indices = selection_matrix.topk(k=1, dim=1)
        neg_dist = torch.gather(neg_dist_matrix, 1, neg_indices).squeeze(1)
        
        # Compute triplet loss with margin
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        
        # Calculate metrics for monitoring
        num_active = torch.sum(loss > 0).item()
        
        components = {
            'mean_pos_dist': pos_dist.mean().item(),
            'mean_neg_dist': neg_dist.mean().item(),
            'num_active_triplets': num_active,
            'triplet_ratio': num_active / batch_size,
            'mining_difficulty': difficulty,
        }
        
        return loss.mean(), components
    
class InfoNCELoss(BaseLoss):
    """
    InfoNCE (Information Noise Contrastive Estimation) loss for contrastive learning.
    
    This loss function is commonly used in self-supervised learning and contrastive
    learning frameworks. It maximizes agreement between differently augmented views
    of the same data while minimizing agreement between views of different data.
    
    Args:
        temperature (float): Temperature parameter for scaling similarities.
            Lower values make the model more confident, higher values make it
            more uncertain. Typical values range from 0.1 to 1.0.
    
    Attributes:
        temperature (float): Temperature scaling parameter
    """
    
    def __init__(self, temperature: float) -> None:
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self, 
        anchors: torch.Tensor, 
        positives: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute InfoNCE loss.
        
        Args:
            anchors (torch.Tensor): Anchor embeddings of shape (N, D)
            positives (torch.Tensor): Positive embeddings of shape (N, D)
            
        Returns:
            Tuple[torch.Tensor, Dict[str, float]]:
                - Loss tensor (scalar)
                - Dictionary containing similarity statistics:
                    - mean_similarity: Average similarity across all pairs
                    - min_similarity: Minimum similarity value
                    - max_similarity: Maximum similarity value
                    - similarity_std: Standard deviation of similarities
        """
        # Normalize features for cosine similarity computation
        anchors = F.normalize(anchors, dim=1)
        positives = F.normalize(positives, dim=1)
        
        # Calculate similarity matrix (cosine similarities)
        similarity_matrix = torch.matmul(anchors, positives.T)
        
        # Calculate components for monitoring
        components = {
            'mean_similarity': similarity_matrix.mean().item(),
            'min_similarity': similarity_matrix.min().item(),
            'max_similarity': similarity_matrix.max().item(),
            'similarity_std': similarity_matrix.std().item(),
        }
        
        # Scale by temperature parameter
        logits = similarity_matrix / self.temperature
        
        # Labels are diagonal elements (positive pairs)
        labels = torch.arange(len(anchors), device=anchors.device)
        
        return F.cross_entropy(logits, labels), components

class BCELoss(BaseLoss):
    """
    Binary Cross-Entropy loss for similarity learning.
    
    This loss treats the similarity learning problem as binary classification,
    where positive pairs should have high similarity (label=1) and negative
    pairs should have low similarity (label=0). It can work with explicit
    negatives or use in-batch negatives.
    
    Args:
        threshold (float, optional): Similarity threshold for classification.
            Currently used for potential future extensions. Defaults to 0.5.
    
    Attributes:
        threshold (float): Similarity threshold
        bce (nn.BCEWithLogitsLoss): Binary cross-entropy loss function
    """
    
    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold = threshold
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(
        self, 
        anchors: torch.Tensor, 
        positives: torch.Tensor, 
        negatives: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute binary cross-entropy loss for similarity learning.
        
        Args:
            anchors (torch.Tensor): Anchor embeddings of shape (N, D)
            positives (torch.Tensor): Positive embeddings of shape (N, D)
            negatives (torch.Tensor, optional): Negative embeddings of shape (N, D).
                If None, uses in-batch negatives. Defaults to None.
                
        Returns:
            Tuple[torch.Tensor, Dict[str, float]]:
                - Loss tensor (scalar)
                - Dictionary containing loss components:
                    - pos_similarity: Average positive pair similarity
                    - pos_loss: Loss for positive pairs
                    - neg_similarity: Average negative pair similarity
                    - neg_loss: Loss for negative pairs
                    - total_loss: Combined loss value
        """
        # Normalize features for cosine similarity computation
        anchors = F.normalize(anchors, dim=1)
        positives = F.normalize(positives, dim=1)
        
        # Calculate similarities for positive pairs
        pos_similarities = F.cosine_similarity(anchors, positives)
        pos_labels = torch.ones_like(pos_similarities)
        pos_loss = self.bce(pos_similarities, pos_labels)
        
        if negatives is not None:
            # If explicit negatives are provided, use them
            negatives = F.normalize(negatives, dim=1)
            neg_similarities = F.cosine_similarity(anchors, negatives)
            neg_labels = torch.zeros_like(neg_similarities)
            neg_loss = self.bce(neg_similarities, neg_labels)
            loss = (pos_loss + neg_loss) / 2
        else:
            # Otherwise, use all non-matching pairs in the batch as negatives
            similarity_matrix = torch.matmul(anchors, positives.T)
            negative_mask = ~torch.eye(len(anchors), dtype=torch.bool, device=anchors.device)
            neg_similarities = similarity_matrix[negative_mask]
            neg_labels = torch.zeros_like(neg_similarities)
            neg_loss = self.bce(neg_similarities, neg_labels)
            loss = (pos_loss + neg_loss) / 2
            
        # Calculate components for logging
        with torch.no_grad():
            components = {
                'pos_similarity': pos_similarities.mean().item(),
                'pos_loss': pos_loss.item(),
                'neg_similarity': neg_similarities.mean().item(),
                'neg_loss': neg_loss.item(),
                'total_loss': loss.item()
            }
            
        return loss, components