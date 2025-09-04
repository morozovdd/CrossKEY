"""
Optimized KNN-based descriptor matcher for medical image correspondence.

This module provides a clean, efficient KNN matcher that integrates seamlessly
with the Descriptor Lightning module without hardcoded assumptions.
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


class KNNMatcher:
    """
    Optimized KNN-based descriptor matcher with flexible evaluation capabilities.
    
    This class provides efficient nearest neighbor matching with support for
    various filtering strategies and evaluation metrics. It's designed to work
    seamlessly with PyTorch tensors and Lightning modules.
    
    Args:
        k: Number of nearest neighbors to find
        distance_threshold: Maximum distance for valid matches
        ratio_threshold: Lowe's ratio test threshold (distance ratio between 1st and 2nd neighbor)
        mutual: Whether to enforce mutual nearest neighbor constraint
        metric: Distance metric ('cosine', 'euclidean', 'manhattan', etc.)
        evaluation_threshold: Distance threshold for considering matches correct in evaluation
    """
    
    def __init__(
        self,
        k: int = 1,
        distance_threshold: Optional[float] = None,
        ratio_threshold: Optional[float] = None,
        mutual: bool = True,
        metric: str = 'cosine',
        evaluation_threshold: float = 5.0
    ) -> None:
        self.k = k
        self.distance_threshold = distance_threshold
        self.ratio_threshold = ratio_threshold
        self.mutual = mutual
        self.metric = metric
        self.evaluation_threshold = evaluation_threshold
        
        # Determine how many neighbors to actually query
        self.k_query = k + 1 if ratio_threshold is not None else k
        
        # Pre-configure the nearest neighbor searcher
        self._nn_params = {
            'n_neighbors': self.k_query,
            'metric': metric,
            'algorithm': 'auto'
        }
    
    def _to_numpy(self, tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert tensor to numpy array if needed."""
        if torch.is_tensor(tensor):
            return tensor.detach().cpu().numpy()
        return tensor
    
    def _apply_ratio_test(
        self, 
        distances: np.ndarray, 
        indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Lowe's ratio test to filter ambiguous matches.
        
        Args:
            distances: Distance array of shape [N, k+1]
            indices: Index array of shape [N, k+1]
            
        Returns:
            Filtered distances and indices of shape [N, k]
        """
        if self.ratio_threshold is None or distances.shape[1] < 2:
            return distances[:, :self.k], indices[:, :self.k]
        
        # Compute ratio between first and second nearest neighbors
        ratios = distances[:, 0] / (distances[:, 1] + 1e-8)  # Add small epsilon for numerical stability
        
        # Apply ratio test
        valid_mask = ratios < self.ratio_threshold
        
        # Initialize output arrays
        filtered_distances = distances[:, :self.k].copy()
        filtered_indices = indices[:, :self.k].copy()
        
        # Mark invalid matches
        filtered_distances[~valid_mask] = np.inf
        filtered_indices[~valid_mask] = -1
        
        return filtered_distances, filtered_indices
    
    def _apply_mutual_constraint(
        self,
        distances: np.ndarray,
        indices: np.ndarray,
        src_desc: np.ndarray,
        tgt_desc: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply mutual nearest neighbor constraint.
        
        Args:
            distances: Forward distances [N, k]
            indices: Forward indices [N, k]
            src_desc: Source descriptors [N, D]
            tgt_desc: Target descriptors [M, D]
            
        Returns:
            Filtered distances and indices
        """
        if not self.mutual:
            return distances, indices
        
        # Compute reverse nearest neighbors
        reverse_nn = NearestNeighbors(**self._nn_params)
        reverse_nn.fit(src_desc)
        rev_distances, rev_indices = reverse_nn.kneighbors(tgt_desc)
        
        # Apply ratio test to reverse matches if needed
        if self.ratio_threshold is not None:
            rev_distances, rev_indices = self._apply_ratio_test(rev_distances, rev_indices)
        
        # Check mutual constraint
        filtered_distances = distances.copy()
        filtered_indices = indices.copy()
        
        for i in range(len(indices)):
            for j in range(self.k):
                tgt_idx = indices[i, j]
                if tgt_idx == -1:
                    continue
                    
                # Check if source point i is among the nearest neighbors of target point tgt_idx
                is_mutual = i in rev_indices[tgt_idx, :self.k]
                
                if not is_mutual:
                    filtered_distances[i, j] = np.inf
                    filtered_indices[i, j] = -1
        
        return filtered_distances, filtered_indices
    
    def _apply_distance_threshold(
        self,
        distances: np.ndarray,
        indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply distance threshold filtering."""
        if self.distance_threshold is None:
            return distances, indices
        
        filtered_distances = distances.copy()
        filtered_indices = indices.copy()
        
        # Mark matches exceeding threshold as invalid
        invalid_mask = distances > self.distance_threshold
        filtered_distances[invalid_mask] = np.inf
        filtered_indices[invalid_mask] = -1
        
        return filtered_distances, filtered_indices
    
    def find_matches(
        self,
        src_descriptors: Union[torch.Tensor, np.ndarray],
        tgt_descriptors: Union[torch.Tensor, np.ndarray],
        return_distances: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Find matches between source and target descriptors.
        
        Args:
            src_descriptors: Source descriptors [N, D]
            tgt_descriptors: Target descriptors [M, D]
            return_distances: Whether to return distances along with indices
            
        Returns:
            If return_distances=True:
                Tuple of (indices, distances) where:
                - indices: Match indices array [N, k], -1 indicates no match
                - distances: Match distances array [N, k], inf indicates no match
            If return_distances=False:
                - indices: Match indices array [N, k]
        """
        # Convert to numpy arrays
        src_np = self._to_numpy(src_descriptors)
        tgt_np = self._to_numpy(tgt_descriptors)
        
        # Fit KNN on target descriptors and find matches
        nn = NearestNeighbors(**self._nn_params)
        nn.fit(tgt_np)
        distances, indices = nn.kneighbors(src_np)
        
        # Apply filtering steps in sequence
        distances, indices = self._apply_ratio_test(distances, indices)
        distances, indices = self._apply_mutual_constraint(distances, indices, src_np, tgt_np)
        distances, indices = self._apply_distance_threshold(distances, indices)
        
        if return_distances:
            return indices, distances
        return indices
    
    def get_match_pairs(
        self,
        src_descriptors: Union[torch.Tensor, np.ndarray],
        tgt_descriptors: Union[torch.Tensor, np.ndarray]
    ) -> List[Tuple[int, int, float]]:
        """
        Get matches as a list of (source_idx, target_idx, distance) tuples.
        
        Args:
            src_descriptors: Source descriptors [N, D]
            tgt_descriptors: Target descriptors [M, D]
            
        Returns:
            List of (src_idx, tgt_idx, distance) tuples for valid matches
        """
        indices, distances = self.find_matches(src_descriptors, tgt_descriptors, return_distances=True)
        
        matches = []
        for src_idx in range(len(indices)):
            for k_idx in range(self.k):
                tgt_idx = indices[src_idx, k_idx]
                if tgt_idx != -1:  # Valid match
                    distance = float(distances[src_idx, k_idx])
                    matches.append((src_idx, tgt_idx, distance))
        
        return matches
    
    def evaluate_matches(
        self,
        src_points: Union[torch.Tensor, np.ndarray],
        tgt_points: Union[torch.Tensor, np.ndarray],
        match_pairs: List[Tuple[int, int, float]]
    ) -> Dict[str, float]:
        """
        Evaluate matching quality based on spatial distances.
        
        Args:
            src_points: Source 3D points [N, 3]
            tgt_points: Target 3D points [M, 3]
            match_pairs: List of (src_idx, tgt_idx, descriptor_distance) tuples
            
        Returns:
            Dictionary containing evaluation metrics:
                - num_matches: Total number of matches
                - num_correct: Number of spatially correct matches
                - precision: Percentage of correct matches
                - matching_score: Ratio of matched to total source points
        """
        if len(match_pairs) == 0:
            return {
                'num_matches': 0,
                'num_correct': 0,
                'precision': 0.0,
                'matching_score': 0.0
            }
        
        # Convert to tensors for easier computation
        src_points = torch.tensor(self._to_numpy(src_points), dtype=torch.float32)
        tgt_points = torch.tensor(self._to_numpy(tgt_points), dtype=torch.float32)
        
        # Extract matched points
        src_indices = [pair[0] for pair in match_pairs]
        tgt_indices = [pair[1] for pair in match_pairs]
        
        matched_src_points = src_points[src_indices]
        matched_tgt_points = tgt_points[tgt_indices]
        
        # Compute spatial distances between matched points
        spatial_distances = torch.norm(matched_src_points - matched_tgt_points, dim=1)
        
        # Count correct matches based on evaluation threshold
        correct_matches = (spatial_distances <= self.evaluation_threshold).sum().item()
        
        # Compute statistics
        metrics = {
            'num_matches': len(match_pairs),
            'num_correct': correct_matches,
            'precision': (correct_matches / len(match_pairs)) * 100.0,
            'matching_score': len(match_pairs) / len(src_points)
        }
        
        return metrics
    
    def match_and_evaluate(
        self,
        src_descriptors: Union[torch.Tensor, np.ndarray],
        tgt_descriptors: Union[torch.Tensor, np.ndarray],
        src_points: Union[torch.Tensor, np.ndarray],
        tgt_points: Union[torch.Tensor, np.ndarray]
    ) -> Tuple[List[Tuple[int, int, float]], Dict[str, float]]:
        """
        Perform matching and evaluation in one step.
        
        Args:
            src_descriptors: Source descriptors [N, D]
            tgt_descriptors: Target descriptors [M, D]
            src_points: Source 3D points [N, 3]
            tgt_points: Target 3D points [M, 3]
            
        Returns:
            Tuple of:
                - List of (src_idx, tgt_idx, descriptor_distance) match pairs
                - Dictionary of evaluation metrics
        """
        match_pairs = self.get_match_pairs(src_descriptors, tgt_descriptors)
        metrics = self.evaluate_matches(src_points, tgt_points, match_pairs)
        return match_pairs, metrics
    
    def get_match_coordinates(
        self,
        src_points: Union[torch.Tensor, np.ndarray],
        tgt_points: Union[torch.Tensor, np.ndarray],
        match_pairs: List[Tuple[int, int, float]]
    ) -> np.ndarray:
        """
        Get matched point coordinates in format suitable for external evaluation.
        
        Args:
            src_points: Source 3D points [N, 3]
            tgt_points: Target 3D points [M, 3]
            match_pairs: List of (src_idx, tgt_idx, distance) tuples
            
        Returns:
            Array of shape [num_matches, 6] with columns [x_src, y_src, z_src, x_tgt, y_tgt, z_tgt]
        """
        if len(match_pairs) == 0:
            return np.empty((0, 6))
        
        src_points = self._to_numpy(src_points)
        tgt_points = self._to_numpy(tgt_points)
        
        matches_array = np.zeros((len(match_pairs), 6))
        
        for i, (src_idx, tgt_idx, _) in enumerate(match_pairs):
            matches_array[i, :3] = src_points[src_idx]
            matches_array[i, 3:] = tgt_points[tgt_idx]
        
        return matches_array