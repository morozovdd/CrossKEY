"""Descriptor extraction and matching for CrossKEY HF Space.

Provides functions for:
1. Re-running KNN matching with new parameters (CPU, fast)
2. Full inference from uploaded volumes + checkpoint (GPU)
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from src.data.datamodule import DescriptorDataModule
from src.model.descriptor import Descriptor
from src.model.matcher import KNNMatcher
from src.utils.utils import load_nifti

from visualization import downsample_volume


def load_precomputed(precomputed_dir: str = "precomputed") -> dict:
    """Load all pre-computed data for the default demo tab.

    Returns:
        Dict with keys: descriptors_mr, descriptors_us, points_mr, points_us,
                        volume_mr, volume_us, metadata
    """
    d = Path(precomputed_dir)
    with open(d / "metadata.json") as f:
        metadata = json.load(f)

    return {
        "descriptors_mr": torch.load(d / "descriptors_mr.pt", weights_only=True),
        "descriptors_us": torch.load(d / "descriptors_us.pt", weights_only=True),
        "points_mr": torch.load(d / "points_mr.pt", weights_only=True).numpy(),
        "points_us": torch.load(d / "points_us.pt", weights_only=True).numpy(),
        "volume_mr": np.load(d / "volume_mr.npy"),
        "volume_us": np.load(d / "volume_us.npy"),
        "metadata": metadata,
    }


def run_matching(
    descriptors_mr: torch.Tensor,
    descriptors_us: torch.Tensor,
    points_mr: np.ndarray,
    points_us: np.ndarray,
    ratio_threshold: float = 0.75,
    mutual: bool = True,
    metric: str = "euclidean",
    evaluation_threshold: float = 5.0,
) -> Tuple[List[Tuple[int, int, float]], Dict[str, float]]:
    """Run KNN matching with given parameters. CPU-only, fast (<1s).

    Returns:
        (match_pairs, metrics) -- same format as KNNMatcher.match_and_evaluate()
    """
    matcher = KNNMatcher(
        k=1,
        distance_threshold=float("inf"),
        ratio_threshold=ratio_threshold,
        mutual=mutual,
        metric=metric,
        evaluation_threshold=evaluation_threshold,
    )
    return matcher.match_and_evaluate(
        descriptors_mr, descriptors_us, points_mr, points_us,
    )


def run_inference(
    mr_path: str,
    us_path: str,
    heatmap_path: str,
    checkpoint_path: str,
    batch_size: int = 64,
    grid_spacing: int = 8,
) -> dict:
    """Run full inference on uploaded volumes. Requires GPU.

    Args:
        mr_path: Path to uploaded MR NIfTI file.
        us_path: Path to uploaded US NIfTI file.
        heatmap_path: Path to uploaded heatmap NIfTI file.
        checkpoint_path: Path to uploaded checkpoint.
        batch_size: Inference batch size.
        grid_spacing: Grid spacing for US keypoint generation.

    Returns:
        Dict with same keys as load_precomputed().
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = Descriptor.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)

    # Create datamodule with custom paths
    dm = DescriptorDataModule(
        data_dir=".",  # Not used when paths are specified
        batch_size=batch_size,
        num_workers=0,
        patch_size=(32, 32, 32),
        grid_spacing=grid_spacing,
        mr_path=mr_path,
        us_path=us_path,
        heatmap_path=heatmap_path,
    )
    dm.setup(stage="test")

    # Extract descriptors
    all_desc, all_pts, all_mod = [], [], []
    with torch.no_grad():
        for batch in dm.test_dataloader():
            desc = model(batch["patch"].to(device))
            all_desc.append(desc.cpu())
            all_pts.append(batch["point"].cpu())
            all_mod.extend(batch["modality"])

    all_desc = torch.cat(all_desc)
    all_pts = torch.cat(all_pts)
    mr_mask = torch.tensor([m == "mr" for m in all_mod])

    # Downsample volumes for rendering
    mr_vol = load_nifti(mr_path)
    us_vol = load_nifti(us_path)
    mr_norm = (mr_vol - mr_vol.min()) / (mr_vol.max() - mr_vol.min() + 1e-8)
    us_norm = (us_vol - us_vol.min()) / (us_vol.max() - us_vol.min() + 1e-8)

    return {
        "descriptors_mr": all_desc[mr_mask],
        "descriptors_us": all_desc[~mr_mask],
        "points_mr": all_pts[mr_mask].numpy(),
        "points_us": all_pts[~mr_mask].numpy(),
        "volume_mr": downsample_volume(mr_norm),
        "volume_us": downsample_volume(us_norm),
        "metadata": {
            "padded_shape_mr": list(dm._mr_volume.shape),
            "padded_shape_us": list(dm._us_volume.shape),
        },
    }
