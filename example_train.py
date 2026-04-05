import argparse
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path

# MPS (Apple Silicon) does not support 3D pooling ops; enable CPU fallback
if platform.system() == 'Darwin':
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from src.model.descriptor import Descriptor
from src.data.datamodule import DescriptorDataModule

logger = logging.getLogger("crosskey.train")


def parse_args():
    parser = argparse.ArgumentParser(description="Train CrossKEY descriptor model")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml",
                        help="Path to training config file")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Path to data directory")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Log output directory (overrides config)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging")
    return parser.parse_args()


def check_and_prepare_data(data_dir: str):
    """Check if required data exists and run preprocessing scripts if needed."""
    logger.info("Checking data availability...")

    sift_output_dir = Path(data_dir) / "sift_output"
    sift_mr_files = list((sift_output_dir / "mr").glob("*_desc.csv")) if (sift_output_dir / "mr").exists() else []
    sift_us_files = list((sift_output_dir / "synthetic_us").glob("*_desc.csv")) if (sift_output_dir / "synthetic_us").exists() else []

    heatmap_dir = Path(data_dir) / "heatmap"
    heatmap_files = list(heatmap_dir.glob("*.nii.gz")) if heatmap_dir.exists() else []

    if not sift_mr_files or not sift_us_files:
        logger.info("SIFT descriptors not found. Running extraction...")
        try:
            result = subprocess.run(
                [sys.executable, "scripts/run_sift.py",
                 "--input-dir", str(Path(data_dir) / "img"),
                 "--output-dir", str(sift_output_dir)],
                check=True, capture_output=True, text=True,
            )
            logger.info("SIFT extraction completed")
            if result.stdout:
                logger.debug(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error("SIFT extraction failed: %s", e.stderr)
            sys.exit(1)
    else:
        logger.info("SIFT descriptors found")

    if not heatmap_files:
        logger.info("Heatmaps not found. Running generation...")
        try:
            result = subprocess.run(
                [sys.executable, "scripts/create_heatmaps.py",
                 "--data-dir", str(Path(data_dir) / "img"),
                 "--output-dir", str(heatmap_dir)],
                check=True, capture_output=True, text=True,
            )
            logger.info("Heatmap generation completed")
            if result.stdout:
                logger.debug(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error("Heatmap generation failed: %s", e.stderr)
            sys.exit(1)
    else:
        logger.info("Heatmaps found")

    logger.info("All required data is ready")


def create_model(config):
    """Create model from configuration."""
    return Descriptor(
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


def create_datamodule(config, data_dir: str):
    """Create datamodule from configuration."""
    data_config = config.get('data', {})
    return DescriptorDataModule(
        data_dir=data_dir,
        batch_size=data_config.get('batch_size', 256),
        num_workers=data_config.get('num_workers', 4),
        patch_size=(data_config.get('patch_size', 32),) * 3,
        num_samples=data_config.get('num_samples', 1024),
        grid_spacing=data_config.get('grid_spacing', 8),
        augment=data_config.get('augment', True),
        max_angle=data_config.get('max_angle', 45.0),
        initial_angle=data_config.get('initial_angle', 5.0),
        angle_warmup_epochs=data_config.get('angle_warmup_epochs', 1000),
    )


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    pl.seed_everything(42)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')

    check_and_prepare_data(args.data_dir)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    logger.info("Configuration loaded from %s", args.config)

    model = create_model(config)
    datamodule = create_datamodule(config, args.data_dir)
    logger.info("Model created with %s parameters", f"{sum(p.numel() for p in model.parameters()):,}")

    log_dir = args.log_dir or config.get('logger', {}).get('save_dir', 'logs/')
    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        name=config.get('logger', {}).get('name', 'descriptor_experiment'),
    )

    trainer_config = config.get('trainer', {})
    trainer = pl.Trainer(
        max_epochs=trainer_config.get('max_epochs', 2000),
        accelerator=trainer_config.get('accelerator', 'auto'),
        devices=trainer_config.get('devices', 'auto'),
        precision=trainer_config.get('precision', 32),
        logger=tb_logger,
        log_every_n_steps=trainer_config.get('log_every_n_steps', 50),
    )

    logger.info("Starting training...")
    try:
        trainer.fit(model, datamodule)
        logger.info("Training completed")
        if hasattr(trainer, 'checkpoint_callback') and trainer.checkpoint_callback:
            if trainer.checkpoint_callback.best_model_path:
                logger.info("Best checkpoint: %s", trainer.checkpoint_callback.best_model_path)
        if trainer.ckpt_path:
            logger.info("Last checkpoint: %s", trainer.ckpt_path)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error("Training failed: %s", e)
        raise


if __name__ == "__main__":
    main()
