"""
Fashion item classifier using MobileNet backbone.

This module implements the complete classifier for fashion items.
"""

import torch
import torch.nn as nn
from typing import Optional

from .backbone import MobileNetBackbone


class MobileNetLikeClassifier(nn.Module):
    """
    A classifier model that combines a feature extraction backbone with a simple classification head.
    
    Architecture:
        Input → Backbone → AdaptiveAvgPool → Flatten → Linear → Logits
        
    Example:
        >>> model = MobileNetLikeClassifier(num_classes=7)
        >>> x = torch.randn(4, 3, 64, 64)
        >>> logits = model(x)
        >>> print(logits.shape)  # torch.Size([4, 7])
    """
    
    def __init__(self, num_classes: int = 7):
        """
        Initializes the classifier components.

        Args:
            num_classes (int): The number of output classes for the final classification layer.
        """
        super().__init__()

        # Backbone extracts features from input images
        self.backbone = MobileNetBackbone()

        # Head processes the features to produce class predictions
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),      # Reduce spatial dimensions to 1x1
            nn.Flatten(),                  # Flatten the features into a 1D vector
            nn.Linear(64, num_classes),    # Map the flattened features to output classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the classifier.

        Args:
            x (torch.Tensor): The input tensor (batch of images).

        Returns:
            torch.Tensor: The raw, unnormalized output scores (logits) for each class.
        """
        # Pass the input through the feature extraction backbone
        x = self.backbone(x)
        # Pass the features through the classification head
        x = self.head(x)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the backbone without classification.
        
        Useful for transfer learning and feature visualization.
        
        Args:
            x (torch.Tensor): The input tensor (batch of images).
            
        Returns:
            torch.Tensor: Feature maps from the backbone [B, 64, H, W].
        """
        return self.backbone(x)


if __name__ == "__main__":
    # Test the MobileNetLikeClassifier
    print("Testing MobileNetLikeClassifier...")
    
    # Test case 1: Standard classification
    model = MobileNetLikeClassifier(num_classes=7)
    x = torch.randn(8, 3, 64, 64)
    logits = model(x)
    print(f"Test 1 - Input: {x.shape}, Output: {logits.shape}")
    assert logits.shape == (8, 7), "Test 1 failed!"
    
    # Test case 2: Different number of classes
    model2 = MobileNetLikeClassifier(num_classes=10)
    logits2 = model2(x)
    print(f"Test 2 - Input: {x.shape}, Output (10 classes): {logits2.shape}")
    assert logits2.shape == (8, 10), "Test 2 failed!"
    
    # Test case 3: Feature extraction
    features = model.get_features(x)
    print(f"Test 3 - Input: {x.shape}, Features: {features.shape}")
    assert features.shape == (8, 64, 4, 4), "Test 3 failed!"
    
    # Test case 4: Single image
    x_single = torch.randn(1, 3, 64, 64)
    logits_single = model(x_single)
    print(f"Test 4 - Single image: {x_single.shape}, Output: {logits_single.shape}")
    assert logits_single.shape == (1, 7), "Test 4 failed!"
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    head_params = sum(p.numel() for p in model.head.parameters())
    
    print(f"\nParameter breakdown:")
    print(f"Backbone: {backbone_params:,}")
    print(f"Head: {head_params:,}")
    print(f"Total: {total_params:,}")
    
    print("✅ All tests passed!")
