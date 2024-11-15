"""
Siamese network implementation for visual similarity learning.

This module implements the Siamese encoder and network for metric learning.
"""

import torch
import torch.nn as nn
from typing import Tuple

from .backbone import MobileNetBackbone


class SiameseEncoder(nn.Module):
    """
    Implements an encoder module suitable for Siamese networks.

    This class takes a pre-defined backbone (feature extractor) and adds a
    representation head (pooling + flatten) to produce a fixed-size vector
    embedding for an input image.
    
    Architecture:
        Input → Backbone → AdaptiveAvgPool → Flatten → Embedding
        
    Example:
        >>> from models.backbone import MobileNetBackbone
        >>> backbone = MobileNetBackbone()
        >>> encoder = SiameseEncoder(backbone)
        >>> x = torch.randn(4, 3, 64, 64)
        >>> embedding = encoder(x)
        >>> print(embedding.shape)  # torch.Size([4, 64])
    """

    def __init__(self, backbone: nn.Module):
        """
        Initializes the SiameseEncoder.

        Args:
            backbone (nn.Module): The convolutional neural network to use
                                  as the feature extractor.
        """
        super(SiameseEncoder, self).__init__()

        # Store the provided backbone model
        self.backbone = backbone

        # Define the representation head
        self.representation = nn.Sequential(
            # Apply adaptive average pooling to reduce spatial dimensions to 1x1
            nn.AdaptiveAvgPool2d(1),
            # Flatten the 1x1 feature map into a 1D vector
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass through the encoder.

        Args:
            x (torch.Tensor): The input tensor (e.g., a batch of images).

        Returns:
            torch.Tensor: The final 1D embedding vector for the input.
        """
        # Extract features using the backbone
        features = self.backbone(x)
        # Convert feature map to a fixed-size vector using the representation head
        representation = self.representation(features)
        
        return representation


class SiameseNetwork(nn.Module):
    """
    Implements the main Siamese Network structure.

    This network takes multiple inputs (anchor, positive, negative during training)
    and processes each through a shared `embedding_network` (the SiameseEncoder)
    to produce corresponding embeddings.
    
    The key principle: All images are processed through the SAME network with
    SHARED weights, ensuring consistent embedding space.
    
    Example:
        >>> encoder = SiameseEncoder(backbone)
        >>> siamese = SiameseNetwork(encoder)
        >>> anchor = torch.randn(4, 3, 64, 64)
        >>> positive = torch.randn(4, 3, 64, 64)
        >>> negative = torch.randn(4, 3, 64, 64)
        >>> a_emb, p_emb, n_emb = siamese(anchor, positive, negative)
        >>> print(a_emb.shape, p_emb.shape, n_emb.shape)  # All [4, 64]
    """

    def __init__(self, embedding_network: nn.Module):
        """
        Initializes the SiameseNetwork.

        Args:
            embedding_network (nn.Module): The shared encoder network (e.g., SiameseEncoder)
                                           that generates embeddings from images.
        """
        super(SiameseNetwork, self).__init__()
        
        # Store the shared embedding network
        self.embedding_network = embedding_network

    def forward(
        self, 
        anchor: torch.Tensor, 
        positive: torch.Tensor, 
        negative: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Defines the forward pass for training with image triplets.

        Args:
            anchor (torch.Tensor): The batch of anchor images.
            positive (torch.Tensor): The batch of positive images (same class as anchor).
            negative (torch.Tensor): The batch of negative images (different class from anchor).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the 
                embeddings for anchor, positive, and negative images.
        """
        # Process the anchor image through the embedding network
        anchor_output = self.embedding_network(anchor)
        # Process the positive image through the *same* embedding network (shared weights)
        positive_output = self.embedding_network(positive)
        # Process the negative image through the *same* embedding network (shared weights)
        negative_output = self.embedding_network(negative)

        return anchor_output, positive_output, negative_output

    def get_embedding(self, image: torch.Tensor) -> torch.Tensor:
        """
        Generates an embedding for a single input image or batch. 
        Used for inference/retrieval.

        Args:
            image (torch.Tensor): The input image tensor (should include batch dimension).

        Returns:
            torch.Tensor: The embedding vector(s) for the image(s).
        """
        return self.embedding_network(image)


if __name__ == "__main__":
    # Test the Siamese components
    print("Testing SiameseEncoder and SiameseNetwork...")
    
    # Create a backbone
    backbone = MobileNetBackbone()
    
    # Test SiameseEncoder
    print("\n1. Testing SiameseEncoder...")
    encoder = SiameseEncoder(backbone)
    x = torch.randn(8, 3, 64, 64)
    embedding = encoder(x)
    print(f"Input: {x.shape}, Embedding: {embedding.shape}")
    assert embedding.shape == (8, 64), "SiameseEncoder test failed!"
    
    # Test SiameseNetwork
    print("\n2. Testing SiameseNetwork...")
    siamese = SiameseNetwork(encoder)
    
    # Create triplet data
    anchor = torch.randn(4, 3, 64, 64)
    positive = torch.randn(4, 3, 64, 64)
    negative = torch.randn(4, 3, 64, 64)
    
    # Forward pass
    anchor_emb, pos_emb, neg_emb = siamese(anchor, positive, negative)
    print(f"Anchor: {anchor_emb.shape}")
    print(f"Positive: {pos_emb.shape}")
    print(f"Negative: {neg_emb.shape}")
    assert anchor_emb.shape == (4, 64), "Anchor embedding shape incorrect!"
    assert pos_emb.shape == (4, 64), "Positive embedding shape incorrect!"
    assert neg_emb.shape == (4, 64), "Negative embedding shape incorrect!"
    
    # Test get_embedding method
    print("\n3. Testing get_embedding...")
    single_image = torch.randn(1, 3, 64, 64)
    single_embedding = siamese.get_embedding(single_image)
    print(f"Single image: {single_image.shape}, Embedding: {single_embedding.shape}")
    assert single_embedding.shape == (1, 64), "get_embedding test failed!"
    
    # Test that embeddings are different for different inputs
    print("\n4. Testing embedding consistency...")
    img1 = torch.randn(1, 3, 64, 64)
    img2 = torch.randn(1, 3, 64, 64)
    emb1 = encoder(img1)
    emb2 = encoder(img2)
    distance = torch.norm(emb1 - emb2).item()
    print(f"Distance between different images: {distance:.4f}")
    assert distance > 0, "Embeddings should be different for different images!"
    
    # Count parameters
    total_params = sum(p.numel() for p in siamese.parameters())
    print(f"\nTotal parameters in Siamese network: {total_params:,}")
    
    print("✅ All tests passed!")
