"""
Inverted Residual Block implementation.

This module implements the core building block used in MobileNetV2-style architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InvertedResidualBlock(nn.Module):
    """
    Implements an inverted residual block, often used in architectures like MobileNetV2.
    
    This block features an expansion phase (1x1 convolution), a depthwise
    convolution (3x3 convolution), and a projection phase (1x1 convolution).
    It utilizes a residual connection between the input and the output of the projection.
    
    Architecture:
        Input → Expand (1×1) → Depthwise (3×3) → Project (1×1) → + Input → Output
        
    Example:
        >>> block = InvertedResidualBlock(24, 32, stride=2, expansion_factor=3)
        >>> x = torch.randn(1, 24, 56, 56)
        >>> output = block(x)
        >>> print(output.shape)  # torch.Size([1, 32, 28, 28])
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int, 
        expansion_factor: int, 
        shortcut: nn.Module = None
    ):
        """
        Initializes the InvertedResidualBlock module.

        Args:
            in_channels (int): The number of channels in the input tensor.
            out_channels (int): The number of channels in the output tensor.
            stride (int): The stride to be used in the depthwise convolution.
            expansion_factor (int): The factor by which to expand the input channels
                                    in the expansion phase.
            shortcut (nn.Module, optional): An optional module to be used for the 
                                           shortcut connection, typically to match 
                                           dimensions if the stride is > 1 or if 
                                           channel counts differ.
        """
        super().__init__()
        
        # Calculate the number of channels for the intermediate (expanded) representation
        hidden_dim = in_channels * expansion_factor

        # Expansion phase: increases the number of channels
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Depthwise convolution phase: lightweight spatial convolution per channel
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_dim,  # Key: depthwise separable
                bias=False,
            ),
            nn.BatchNorm2d(num_features=hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Projection phase: reduces the number of channels to out_channels
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # Optional shortcut connection for residual learning
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the InvertedResidualBlock.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the block operations
                          and the residual connection, followed by a ReLU activation.
        """
        # Save input for residual connection
        skip = x

        # Forward pass through the block
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.project(out)

        # Apply shortcut if it exists (for matching dimensions)
        if self.shortcut is not None:
            skip = self.shortcut(x)

        # Add the skip connection
        out = out + skip
        
        # Final activation
        return F.relu(out)


if __name__ == "__main__":
    # Test the InvertedResidualBlock
    print("Testing InvertedResidualBlock...")
    
    # Test case 1: No dimension change (stride=1, same channels)
    block1 = InvertedResidualBlock(
        in_channels=24,
        out_channels=24,
        stride=1,
        expansion_factor=3,
        shortcut=None
    )
    x1 = torch.randn(2, 24, 56, 56)
    out1 = block1(x1)
    print(f"Test 1 - Input: {x1.shape}, Output: {out1.shape}")
    assert out1.shape == (2, 24, 56, 56), "Test 1 failed!"
    
    # Test case 2: With stride and channel change
    shortcut = nn.Sequential(
        nn.Conv2d(24, 32, 1, stride=2, bias=False),
        nn.BatchNorm2d(32),
    )
    block2 = InvertedResidualBlock(
        in_channels=24,
        out_channels=32,
        stride=2,
        expansion_factor=6,
        shortcut=shortcut
    )
    x2 = torch.randn(2, 24, 56, 56)
    out2 = block2(x2)
    print(f"Test 2 - Input: {x2.shape}, Output: {out2.shape}")
    assert out2.shape == (2, 32, 28, 28), "Test 2 failed!"
    
    print("✅ All tests passed!")
