import torch
import numpy as np
from pathlib import Path
import yaml
import pytorch_lightning as pl

from src.model.descriptor import Descriptor
from src.data.datamodule import DescriptorDataModule

def create_model(checkpoint, config):
    """Create model from configuration."""
    return Descriptor.load_from_checkpoint(
        checkpoint_path=checkpoint,
        # Model parameters
        out_dim=config.get('model', {}).get('out_dim', 512),
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


def load_model_from_checkpoint(checkpoint_path, config_path="configs/test_config.yaml"):
    """Load model from checkpoint."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model from checkpoint
    model = create_model(checkpoint=checkpoint_path, config=config)

    # Set to evaluation mode
    model.eval()
    return model


def main():
    """Main function to test the model with real US images."""
    print("🔑 CrossKEY - Testing with Real Ultrasound")
    print("=" * 50)
    
    # Paths
    data_dir = Path("data")
    config_path = "configs/test_config.yaml"
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get checkpoint path from config
    checkpoint_path = config.get('model', {}).get('checkpoint_path')
    if not checkpoint_path:
        print("❌ Checkpoint path not found in config file")
        print("Please add model.checkpoint_path to configs/test_config.yaml")
        return
    
    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        print("Please train the model first or provide the correct checkpoint path.")
        return
    
    # Check if config exists
    if not Path(config_path).exists():
        print(f"❌ Config file not found at {config_path}")
        return
    
    # Check if data directory exists
    if not data_dir.exists():
        print(f"❌ Data directory not found at {data_dir}")
        return
    
    try:
        # Load config
        print("📋 Loading configuration...")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create datamodule for testing
        print("📦 Setting up data module...")
        datamodule = DescriptorDataModule(
            data_dir=str(data_dir),
            batch_size=config.get('data', {}).get('batch_size', 256),
            num_workers=config.get('data', {}).get('num_workers', 0),
            patch_size=(config.get('patch_size', 32),) * 3,
            grid_spacing=config.get('data', {}).get('grid_spacing', 8)
        )
        
        # Setup for testing
        print("� Setting up data for testing...")
        datamodule.setup(stage='test')
        
        # Get test dataloader
        print("📥 Creating test dataloader...")
        test_dataloader = datamodule.test_dataloader()
        print(f"✅ Test dataloader created with {len(test_dataloader)} batches")
        
        # Load model
        print("🧠 Loading model from checkpoint...")
        model = load_model_from_checkpoint(checkpoint_path, config_path)
        print("✅ Model loaded successfully")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"� Using device: {device}")
        
        # Create Lightning trainer for testing
        trainer = pl.Trainer(
            accelerator='auto',
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            enable_model_summary=False
        )
        
        # Run testing
        print("🚀 Running model testing...")
        results = trainer.test(model, test_dataloader, verbose=True)
        
        print(f"\n✅ Testing completed successfully!")
        print(f"📊 Results: {results}")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Make sure the model checkpoint is compatible and the data is properly formatted.")


if __name__ == "__main__":
    main()
