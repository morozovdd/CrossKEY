import argparse
import logging
import os
import platform
from pathlib import Path

# MPS (Apple Silicon) does not support 3D pooling ops; enable CPU fallback
if platform.system() == 'Darwin':
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import yaml
import torch
import pytorch_lightning as pl

from src.model.descriptor import Descriptor
from src.data.datamodule import DescriptorDataModule

logger = logging.getLogger("crosskey.test")


def parse_args():
    parser = argparse.ArgumentParser(description="Test CrossKEY descriptor model")
    parser.add_argument("--config", type=str, default="configs/test_config.yaml",
                        help="Path to test config file")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Path to data directory")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint path (overrides config)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging")
    return parser.parse_args()


def create_model(checkpoint_path, config):
    """Load model from checkpoint with config overrides."""
    return Descriptor.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        out_dim=config.get('model', {}).get('out_dim', 512),
        input_channels=config.get('model', {}).get('input_channels', 1),
        loss_type=config.get('loss', {}).get('type', 'triplet'),
        margin=config.get('loss', {}).get('margin', 1.0),
        temperature=config.get('loss', {}).get('temperature', 0.1),
        warmup_epochs=config.get('loss', {}).get('warmup_epochs', 200),
        spatial_weight=config.get('loss', {}).get('spatial_weight', 0.5),
        learning_rate=config.get('optimizer', {}).get('learning_rate', 1e-4),
        weight_decay=config.get('optimizer', {}).get('weight_decay', 1e-5),
        max_epochs=config.get('trainer', {}).get('max_epochs', 2000),
        eta_min=config.get('optimizer', {}).get('eta_min', 1e-6),
        knn_k=config.get('evaluation', {}).get('knn_k', 1),
        distance_threshold=config.get('evaluation', {}).get('distance_threshold', float('inf')),
        ratio_threshold=config.get('evaluation', {}).get('ratio_threshold', 0.8),
        mutual=config.get('evaluation', {}).get('mutual', True),
        metric=config.get('evaluation', {}).get('metric', 'euclidean'),
        max_distance=config.get('evaluation', {}).get('max_distance', 5.0),
    )


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    logger.info("Configuration loaded from %s", args.config)

    checkpoint_path = args.checkpoint or config.get('model', {}).get('checkpoint_path')
    if not checkpoint_path:
        logger.error("No checkpoint path provided. Use --checkpoint or set model.checkpoint_path in config.")
        return

    if not Path(checkpoint_path).exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        return

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        return

    data_config = config.get('data', {})
    datamodule = DescriptorDataModule(
        data_dir=str(data_dir),
        batch_size=data_config.get('batch_size', 256),
        num_workers=data_config.get('num_workers', 0),
        patch_size=(data_config.get('patch_size', 32),) * 3,
        grid_spacing=data_config.get('grid_spacing', 8),
    )

    logger.info("Setting up test data...")
    datamodule.setup(stage='test')
    test_dataloader = datamodule.test_dataloader()
    logger.info("Test dataloader: %d batches", len(test_dataloader))

    logger.info("Loading model from %s", checkpoint_path)
    model = create_model(checkpoint_path, config)
    model.eval()

    trainer = pl.Trainer(
        accelerator='auto',
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    logger.info("Running evaluation...")
    results = trainer.test(model, test_dataloader, verbose=True)
    logger.info("Results: %s", results)


if __name__ == "__main__":
    main()
