import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
import torch
import subprocess
import sys
import os

from src.model.descriptor import Descriptor
from src.data.datamodule import DescriptorDataModule


def check_and_prepare_data():
    """Check if required data exists and run preprocessing scripts if needed."""
    print("🔍 Checking data availability...")
    
    # Check if SIFT descriptors exist
    sift_output_dir = Path("data/sift_output")
    sift_mr_files = list((sift_output_dir / "mr").glob("*_desc.csv")) if (sift_output_dir / "mr").exists() else []
    sift_us_files = list((sift_output_dir / "synthetic_us").glob("*_desc.csv")) if (sift_output_dir / "synthetic_us").exists() else []
    
    # Check if heatmaps exist  
    heatmap_dir = Path("data/heatmap")
    heatmap_files = list(heatmap_dir.glob("*.nii.gz")) if heatmap_dir.exists() else []
    
    # Run SIFT extraction if needed
    if not sift_mr_files or not sift_us_files:
        print("📥 SIFT descriptors not found. Running SIFT extraction...")
        try:
            result = subprocess.run([sys.executable, "scripts/run_sift.py"], 
                                 check=True, capture_output=True, text=True)
            print("✅ SIFT extraction completed successfully")
            if result.stdout:
                print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"❌ SIFT extraction failed: {e}")
            print(f"Error output: {e.stderr}")
            sys.exit(1)
        except FileNotFoundError:
            print("❌ scripts/run_sift.py not found!")
            sys.exit(1)
    else:
        print("✅ SIFT descriptors found")
    
    # Run heatmap generation if needed
    if not heatmap_files:
        print("📥 Heatmaps not found. Running heatmap generation...")
        try:
            result = subprocess.run([sys.executable, "scripts/create_heatmaps.py"], 
                                 check=True, capture_output=True, text=True)
            print("✅ Heatmap generation completed successfully")
            if result.stdout:
                print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"❌ Heatmap generation failed: {e}")
            print(f"Error output: {e.stderr}")
            sys.exit(1)
        except FileNotFoundError:
            print("❌ scripts/create_heatmaps.py not found!")
            sys.exit(1)
    else:
        print("✅ Heatmaps found")
    
    print("🎉 All required data is ready for training!")


def load_config(config_path: str = "configs/train_config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_model(config):
    """Create model from configuration."""
    return Descriptor(
        # Model parameters
        out_dim=config.get('model', {}).get('out_dim', 128),
        input_channels=config.get('model', {}).get('input_channels', 1),
        
        # Loss parameters
        loss_type=config.get('loss', {}).get('type', 'triplet'),
        margin=config.get('loss', {}).get('margin', 1.0),
        temperature=config.get('loss', {}).get('temperature', 0.1),
        warmup_epochs=config.get('loss', {}).get('warmup_epochs', 200),
        spatial_weight=config.get('loss', {}).get('spatial_weight', 0.3),
        
        # Optimizer parameters
        learning_rate=config.get('optimizer', {}).get('learning_rate', 1e-4),
        weight_decay=config.get('optimizer', {}).get('weight_decay', 1e-5),
        max_epochs=config.get('trainer', {}).get('max_epochs', 1000),
        eta_min=config.get('optimizer', {}).get('eta_min', 1e-6),
        
        # Evaluation parameters
        knn_k=config.get('evaluation', {}).get('knn_k', 1),
        distance_threshold=config.get('evaluation', {}).get('distance_threshold', float('inf')),
        ratio_threshold=config.get('evaluation', {}).get('ratio_threshold', 0.8),
        mutual=config.get('evaluation', {}).get('mutual', True),
        metric=config.get('evaluation', {}).get('metric', 'euclidean'),
        max_distance=config.get('evaluation', {}).get('max_distance', 5.0),
    )


def create_datamodule(config, data_dir: str = "data"):
    """Create datamodule from configuration."""
    data_config = config.get('data', {})
    
    return DescriptorDataModule(
        data_dir=data_dir,
        batch_size=data_config.get('batch_size', 32),
        num_workers=data_config.get('num_workers', 4),
        patch_size=(data_config.get('patch_size', 32),) * 3,
        num_samples=data_config.get('num_samples', 1024),
        grid_spacing=data_config.get('grid_spacing', 8),
        augment=data_config.get('augment', True),
        max_angle=data_config.get('max_angle', 45.0),
        initial_angle=data_config.get('initial_angle', 5.0),
        angle_warmup_epochs=data_config.get('angle_warmup_epochs', 1000),
    )


# def setup_callbacks(config):
#     """Setup training callbacks."""
#     callbacks = []
    
#     # Early stopping
#     early_stop_config = config.get('early_stopping', {})
#     if early_stop_config.get('enabled', False):
#         early_stop = EarlyStopping(
#             monitor=early_stop_config.get('monitor', 'train/loss'),
#             patience=early_stop_config.get('patience', 50),
#             mode=early_stop_config.get('mode', 'min'),
#             min_delta=early_stop_config.get('min_delta', 0.0001),
#             verbose=False,
#         )
#         callbacks.append(early_stop)
#         print(f"Early stopping enabled with patience {early_stop.patience}")
    
#     # Learning rate monitoring
#     callbacks.append(LearningRateMonitor(logging_interval='epoch'))
    
#     return callbacks


def main():
    """Main training function."""

    pl.seed_everything(42)
    
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
    
    # Check and prepare required data (SIFT descriptors and heatmaps)
    check_and_prepare_data()
    
    # Load configuration
    config = load_config()
    print("Configuration loaded successfully")
    
    # Create model and datamodule
    model = create_model(config)
    datamodule = create_datamodule(config)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Setup callbacks and logger
    # callbacks = setup_callbacks(config)
    logger = TensorBoardLogger(
        save_dir=config.get('logger', {}).get('save_dir', 'logs/'),
        name=config.get('logger', {}).get('name', 'descriptor_experiment'),
    )
    
    # Create trainer
    trainer_config = config.get('trainer', {})
    trainer = pl.Trainer(
        max_epochs=trainer_config.get('max_epochs', 1000),
        accelerator=trainer_config.get('accelerator', 'auto'),
        devices=trainer_config.get('devices', 'auto'),
        precision=trainer_config.get('precision', 32),
        # callbacks=callbacks,
        logger=logger,
        log_every_n_steps=trainer_config.get('log_every_n_steps', 50),
        # gradient_clip_val=trainer_config.get('gradient_clip_val', 1.0),
    )
    
    # Start training
    print("Starting training...")
    try:
        trainer.fit(model, datamodule)
        
        # Print results
        print("Training completed!")
        if hasattr(trainer, 'checkpoint_callback') and trainer.checkpoint_callback:
            if trainer.checkpoint_callback.best_model_path:
                print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
            if trainer.checkpoint_callback.best_model_score is not None:
                print(f"Best loss: {trainer.checkpoint_callback.best_model_score:.6f}")
        
        # Always print the last checkpoint path (Lightning saves this automatically)
        if trainer.ckpt_path:
            print(f"Last checkpoint: {trainer.ckpt_path}")
        else:
            print("No checkpoints were saved")
            
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()