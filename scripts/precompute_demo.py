"""Generate pre-computed data for the HF Space demo.

Runs inference on the example data and saves descriptors, points,
and downsampled volumes for the Space's default "Explore" tab.

Usage:
    python scripts/precompute_demo.py \
        --checkpoint logs/demo_tb/version_0/checkpoints/epoch=1999-step=8000.ckpt \
        --data-dir data \
        --output-dir space/precomputed
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import zoom

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.datamodule import DescriptorDataModule
from src.model.descriptor import Descriptor
from src.utils.utils import load_nifti

logger = logging.getLogger("crosskey.precompute")

TARGET_VOLUME_SIZE = 64


def downsample_volume(volume: np.ndarray, target_size: int = TARGET_VOLUME_SIZE) -> np.ndarray:
    """Downsample volume to approximately target_size^3."""
    factors = [target_size / s for s in volume.shape]
    return zoom(volume, factors, order=1)


def main():
    parser = argparse.ArgumentParser(description="Generate pre-computed data for HF Space")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--output-dir", type=str, default="space/precomputed", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--grid-spacing", type=int, default=8, help="US grid spacing")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info("Loading checkpoint: %s", args.checkpoint)
    model = Descriptor.load_from_checkpoint(args.checkpoint)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Create datamodule for inference
    logger.info("Setting up inference data...")
    dm = DescriptorDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=0,
        patch_size=(32, 32, 32),
        grid_spacing=args.grid_spacing,
    )
    dm.setup(stage="test")

    # Extract descriptors
    logger.info("Extracting descriptors...")
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

    mr_desc = all_desc[mr_mask]
    us_desc = all_desc[~mr_mask]
    mr_pts = all_pts[mr_mask]
    us_pts = all_pts[~mr_mask]

    logger.info("Descriptors: %d MR, %d US (dim=%d)", len(mr_desc), len(us_desc), mr_desc.shape[1])

    # Save descriptors and points
    torch.save(mr_desc, output_dir / "descriptors_mr.pt")
    torch.save(us_desc, output_dir / "descriptors_us.pt")
    torch.save(mr_pts, output_dir / "points_mr.pt")
    torch.save(us_pts, output_dir / "points_us.pt")

    # Load and downsample volumes for visualization
    logger.info("Downsampling volumes for rendering...")
    data_dir = Path(args.data_dir)

    mr_files = list((data_dir / "img" / "mr").glob("*.nii.gz"))
    us_files = list((data_dir / "img" / "us").glob("*.nii.gz"))
    mr_vol = load_nifti(mr_files[0])
    us_vol = load_nifti(us_files[0])

    # Normalize for rendering (simple min-max, display only)
    mr_norm = (mr_vol - mr_vol.min()) / (mr_vol.max() - mr_vol.min() + 1e-8)
    us_norm = (us_vol - us_vol.min()) / (us_vol.max() - us_vol.min() + 1e-8)

    mr_small = downsample_volume(mr_norm)
    us_small = downsample_volume(us_norm)

    np.save(output_dir / "volume_mr.npy", mr_small.astype(np.float32))
    np.save(output_dir / "volume_us.npy", us_small.astype(np.float32))

    # Save metadata (padded shapes for coordinate scaling in visualization)
    padded_shape_mr = list(dm._mr_volume.shape)
    padded_shape_us = list(dm._us_volume.shape)
    metadata = {
        "padded_shape_mr": padded_shape_mr,
        "padded_shape_us": padded_shape_us,
        "num_mr_descriptors": len(mr_desc),
        "num_us_descriptors": len(us_desc),
        "descriptor_dim": int(mr_desc.shape[1]),
        "grid_spacing": args.grid_spacing,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved pre-computed data to %s", output_dir)
    logger.info("Files: %s", [p.name for p in output_dir.iterdir()])


if __name__ == "__main__":
    main()
