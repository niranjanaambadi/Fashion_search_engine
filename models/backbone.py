"""
MobileNet-inspired backbone for feature extraction.

This module implements a simplified MobileNet-like backbone using inverted residual blocks.
"""

import torch
import torch.nn as nn
from typing import Optional

from .blocks import InvertedResidualBlock


class MobileNetBackbone(nn.Module):
    """
    Implements a simplified MobileNet-like backbone feature extractor.

    This class defines the initial stem and a sequence of inverted residual blocks
    to extract features from an input image.
    
    Architecture:
        Input [3, H, W]
          ↓
        Stem: Conv3x3(stride=2) + BN + ReLU → [16, H/2, W/2]
          ↓
        Block1: InvertedResidual(16→24, stride=2) → [24, H/4, W/4]
          ↓
        Block2: InvertedResidual(24→32, stride=2) → [32, H/8, W/8]
          ↓
        Block3: InvertedResidual(32→64, stride=2) → [64, H/16, W/16]
    
    Example:
        >>> backbone = MobileNetBackbone()
        >>> x = torch.randn(1, 3, 64, 64)
        >>> features = backbone(x)
        >>> print(features.shape)  # torch.Size([1, 64, 4, 4])
    """

    def __init__(self):
        """Initializes the layers of the MobileNet backbone."""
        super().__init__()
        
        # Define the initial "stem" convolution layer
        # This layer reduces spatial size and increases channel depth
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # Define the main stack of custom MobileNet-like blocks
        self.blocks = nn.Sequential(
            # Block 1: 16 → 24 channels, spatial /2
            self._make_block(16, 24, stride=2, expansion_factor=3),
            # Block 2: 24 → 32 channels, spatial /2
            self._make_block(24, 32, stride=2, expansion_factor=3),
            # Block 3: 32 → 64 channels, spatial /2
            self._make_block(32, 64, stride=2, expansion_factor=6),
        )

    def _make_block(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1, 
        expansion_factor: int = 6
    ) -> InvertedResidualBlock:
        """
        Helper method to create a single InvertedResidualBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): The stride to be used in the depthwise convolution.
            expansion_factor (int): The factor to expand the channels internally.
            
        Returns:
            InvertedResidualBlock: The created block with optional shortcut.
        """
        # Determine if a shortcut connection is needed
        # A shortcut is needed if input/output channels differ or if stride > 1
        condition = (in_channels != out_channels) or (stride != 1)
        
        if condition:
            # Define the shortcut connection to match dimensions
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            # No shortcut connection is needed
            shortcut = None

        # Instantiate the InvertedResidualBlock
        block = InvertedResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            expansion_factor=expansion_factor,
            shortcut=shortcut
        )

        return block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the backbone.

        Args:
            x (torch.Tensor): The input tensor (e.g., a batch of images).

        Returns:
            torch.Tensor: The output feature map tensor.
        """
        # Pass the input through the initial stem layer
        x = self.stem(x)
        # Pass the result through the main stack of blocks
        x = self.blocks(x)
        
        return x


if __name__ == "__main__":
    # Test the MobileNetBackbone
    print("Testing MobileNetBackbone...")
    
    backbone = MobileNetBackbone()
    
    # Test with different input sizes
    test_cases = [
        (1, 3, 64, 64),   # Single image, 64x64
        (8, 3, 128, 128), # Batch of 8, 128x128
        (16, 3, 224, 224) # Batch of 16, 224x224
    ]
    
    for i, input_shape in enumerate(test_cases, 1):
        x = torch.randn(*input_shape)
        out = backbone(x)
        expected_h = input_shape[2] // 16  # 4 stride-2 operations
        expected_w = input_shape[3] // 16
        expected_shape = (input_shape[0], 64, expected_h, expected_w)
        
        print(f"Test {i} - Input: {x.shape}, Output: {out.shape}")
        assert out.shape == expected_shape, f"Test {i} failed! Expected {expected_shape}, got {out.shape}"
    
    # Count parameters
    total_params = sum(p.numel() for p in backbone.parameters())
    trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("✅ All tests passed!")
