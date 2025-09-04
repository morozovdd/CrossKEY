"""
3D ResNet implementation for feature extraction.

This module provides a 3D ResNet architecture implementation based on the original
ResNet paper, adapted for 3D volumetric data processing. The network is designed
for feature extraction tasks and includes normalization of output features.

Classes:
    BasicBlock3D: Basic building block for 3D ResNet
    ResNet3D: Main 3D ResNet architecture
    
Functions:
    resnet18_3d_feature_extractor: Factory function for ResNet-18 3D model
"""

from typing import Optional, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock3D(nn.Module):
    """
    Basic building block for 3D ResNet.
    
    This class implements the basic residual block for 3D ResNet, consisting of
    two 3D convolutional layers with batch normalization and ReLU activation.
    Includes skip connections for residual learning.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        stride (int, optional): Stride for the first convolution. Defaults to 1.
        downsample (nn.Module, optional): Downsampling layer for skip connection. 
            Defaults to None.
    
    Attributes:
        expansion (int): Channel expansion factor for this block type
    """
    expansion: int = 1
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1, 
        downsample: Optional[nn.Module] = None
    ) -> None:
        super(BasicBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, 
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, 
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the basic block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W)
            
        Returns:
            torch.Tensor: Output tensor after residual block processing
        """
        identity = x

        # First convolution block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second convolution block
        out = self.conv2(out)
        out = self.bn2(out)

        # Apply downsampling to identity if needed
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add residual connection and apply activation
        out += identity
        out = self.relu(out)

        return out

class ResNet3D(nn.Module):
    """
    3D ResNet architecture for feature extraction.
    
    This class implements a 3D ResNet network based on the original ResNet paper,
    adapted for volumetric data. The network outputs normalized feature vectors
    suitable for metric learning tasks.
    
    Args:
        block (nn.Module): The basic block class to use (e.g., BasicBlock3D)
        layers (List[int]): Number of blocks in each layer (4 layers total)
        out_dim (int, optional): Output feature dimension. Defaults to 128.
        input_channels (int, optional): Number of input channels. Defaults to 1.
        zero_init_residual (bool, optional): Whether to zero-initialize the last
            batch norm in each residual block. Defaults to False.
    
    Attributes:
        in_channels (int): Current number of channels (used during layer construction)
        out_dim (int): Output feature dimension
    """
    
    def __init__(
        self, 
        block: nn.Module, 
        layers: List[int], 
        out_dim: int = 128, 
        input_channels: int = 1, 
        zero_init_residual: bool = False
    ) -> None:
        super(ResNet3D, self).__init__()
        
        self.in_channels = 64
        self.out_dim = out_dim
        
        # Initial convolution and pooling
        self.conv1 = nn.Conv3d(
            input_channels, 64, 
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Final pooling and classification
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, out_dim)

        # Initialize weights
        self._initialize_weights()
        
        # Zero-initialize residual connections if requested
        if zero_init_residual:
            self._zero_init_residual_weights(block)

    def _initialize_weights(self) -> None:
        """Initialize network weights using standard initialization schemes."""
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def _zero_init_residual_weights(self, block: nn.Module) -> None:
        """Zero-initialize the last batch norm in each residual block."""
        for module in self.modules():
            if isinstance(module, block):
                nn.init.constant_(module.bn2.weight, 0)

    def _make_layer(
        self, 
        block: nn.Module, 
        out_channels: int, 
        blocks: int, 
        stride: int = 1
    ) -> nn.Sequential:
        """
        Create a layer consisting of multiple residual blocks.
        
        Args:
            block (nn.Module): The block class to use
            out_channels (int): Number of output channels for the layer
            blocks (int): Number of blocks in this layer
            stride (int, optional): Stride for the first block. Defaults to 1.
            
        Returns:
            nn.Sequential: Sequential container of residual blocks
        """
        downsample = None
        
        # Create downsampling layer if needed
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.in_channels, 
                    out_channels * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm3d(out_channels * block.expansion),
            )

        layers = []
        
        # First block (may have stride > 1 and downsampling)
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        # Remaining blocks (stride = 1, no downsampling)
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the 3D ResNet.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W) where:
                - N: batch size
                - C: number of channels
                - D: depth dimension
                - H: height dimension  
                - W: width dimension
                
        Returns:
            torch.Tensor: Normalized feature vector of shape (N, out_dim)
        """
        # Initial convolution and pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Pass through residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling and feature extraction
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        # L2 normalization for metric learning
        x = F.normalize(x, p=2, dim=1)

        return x

def resnet18_3d_encoder(
    input_channels: int = 1, 
    feature_dim: int = 128
) -> ResNet3D:
    """
    Constructs a ResNet-18 model for 3D feature extraction.
    
    This function creates a 3D ResNet-18 architecture with the standard
    layer configuration [2, 2, 2, 2], suitable for feature extraction
    from 3D volumetric data.
    
    Args:
        input_channels (int, optional): Number of input channels. Defaults to 1.
        feature_dim (int, optional): Dimension of output features. Defaults to 128.
        
    Returns:
        ResNet3D: Configured ResNet-18 3D model
        
    Example:
        >>> model = resnet18_3d_encoder(input_channels=1, feature_dim=256)
        >>> x = torch.randn(2, 1, 32, 64, 64)  # (batch, channels, depth, height, width)
        >>> features = model(x)  # Output shape: (2, 256)
    """
    model = ResNet3D(
        BasicBlock3D, 
        layers=[2, 2, 2, 2], 
        out_dim=feature_dim, 
        input_channels=input_channels
    )
    return model