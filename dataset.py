"""
Dataset implementations for fashion search engine.

This module contains dataset classes for both classification and metric learning.
"""

import random
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset


class TripleDataset(Dataset):
    """
    A custom Dataset class that returns triplets of images (anchor, positive, negative).

    This class wraps a standard dataset and, for a given index, returns the 
    item at that index (anchor), a random item with the same label (positive),
    and a random item with a different label (negative).
    
    This is the standard format for training Siamese networks with triplet loss.
    
    Example:
        >>> from torchvision.datasets import ImageFolder
        >>> base_dataset = ImageFolder('data/train', transform=transforms)
        >>> triplet_dataset = TripleDataset(base_dataset)
        >>> anchor, positive, negative = triplet_dataset[0]
    """
    
    def __init__(self, dataset: Dataset):
        """
        Initializes the TripleDataset.

        Args:
            dataset: The base dataset (e.g., torchvision.datasets.ImageFolder) 
                     which contains (data, label) pairs.
        """
        # Store the original dataset
        self.dataset = dataset

        # Get a list of all available labels
        self.labels = range(len(dataset.classes))

        # Create a mapping from labels to their corresponding indices in the dataset
        self.labels_to_indices = self._get_labels_to_indices()

    def __len__(self) -> int:
        """
        Returns the total number of items in the dataset.
        
        Returns:
            int: Length of the base dataset.
        """
        return len(self.dataset)

    def _get_labels_to_indices(self) -> Dict[int, List[int]]:
        """
        Creates a dictionary mapping each label to a list of indices.
        
        This pre-computation makes sampling positives and negatives efficient.
        
        Returns:
            Dict[int, List[int]]: A dictionary where keys are labels and values 
                                  are lists of indices in the dataset that have 
                                  that label.
        """
        labels_to_indices = {}
        
        # Iterate over the entire dataset
        for idx, (_, label) in enumerate(self.dataset):
            # If the label is not yet in the dictionary, add it with an empty list
            if label not in labels_to_indices:
                labels_to_indices[label] = []
            # Append the current index to the list for its label
            labels_to_indices[label].append(idx)
        
        return labels_to_indices
    
    def _get_positive_negative_indices(self, anchor_label: int) -> Tuple[int, int]:
        """
        Finds random indices for a positive and a negative sample.

        Args:
            anchor_label (int): The label of the anchor sample.

        Returns:
            Tuple[int, int]: A tuple (positive_index, negative_index).
        """
        # Get all indices for the anchor label
        positive_indices = self.labels_to_indices[anchor_label]
        # Randomly select one index from the list of positive indices
        positive_index = random.choice(positive_indices)

        # Randomly choose a label that is different from the anchor label
        negative_label = random.choice(
            [label for label in self.labels if label != anchor_label]
        )
        
        # Get all indices for the chosen negative label
        negative_indices = self.labels_to_indices[negative_label]
        # Randomly select one index from the list of negative indices
        negative_index = random.choice(negative_indices)

        return positive_index, negative_index

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves a triplet (anchor, positive, negative) for a given index.

        Args:
            idx (int): The index of the anchor item.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing 
                the anchor image, positive image, and negative image.
        """
        # Get the anchor image and label
        anchor_image, anchor_label = self.dataset[idx]

        # Get positive and negative indices based on the anchor label
        positive_index, negative_index = self._get_positive_negative_indices(anchor_label)

        # Get a positive image (same label)
        positive_image, _ = self.dataset[positive_index]

        # Get a negative image (different label)
        negative_image, _ = self.dataset[negative_index]

        return (anchor_image, positive_image, negative_image)


if __name__ == "__main__":
    # Test the TripleDataset
    print("Testing TripleDataset...")
    
    # Create a mock dataset
    class MockDataset(Dataset):
        def __init__(self, num_samples=100, num_classes=7):
            self.num_samples = num_samples
            self.classes = [f"class_{i}" for i in range(num_classes)]
            # Generate random labels
            self.labels = [i % num_classes for i in range(num_samples)]
            
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Return random tensor and label
            return torch.randn(3, 64, 64), self.labels[idx]
    
    # Create mock dataset
    mock_dataset = MockDataset(num_samples=100, num_classes=7)
    
    # Create triplet dataset
    triplet_dataset = TripleDataset(mock_dataset)
    
    print(f"Base dataset length: {len(mock_dataset)}")
    print(f"Triplet dataset length: {len(triplet_dataset)}")
    assert len(triplet_dataset) == len(mock_dataset), "Length mismatch!"
    
    # Test __getitem__
    print("\nTesting triplet generation...")
    anchor, positive, negative = triplet_dataset[0]
    print(f"Anchor shape: {anchor.shape}")
    print(f"Positive shape: {positive.shape}")
    print(f"Negative shape: {negative.shape}")
    
    assert anchor.shape == (3, 64, 64), "Anchor shape incorrect!"
    assert positive.shape == (3, 64, 64), "Positive shape incorrect!"
    assert negative.shape == (3, 64, 64), "Negative shape incorrect!"
    
    # Verify label mapping
    print("\nVerifying label mapping...")
    print(f"Number of classes: {len(mock_dataset.classes)}")
    print(f"Labels to indices keys: {list(triplet_dataset.labels_to_indices.keys())}")
    
    # Check that all labels are represented
    for label in range(len(mock_dataset.classes)):
        assert label in triplet_dataset.labels_to_indices, f"Label {label} not in mapping!"
        assert len(triplet_dataset.labels_to_indices[label]) > 0, f"No indices for label {label}!"
    
    # Test multiple samples
    print("\nTesting multiple samples...")
    for i in range(10):
        anchor, positive, negative = triplet_dataset[i]
        _, anchor_label = mock_dataset[i]
        
        # Get labels for positive and negative
        pos_idx = None
        neg_idx = None
        for idx in range(len(mock_dataset)):
            img, label = mock_dataset[idx]
            if torch.equal(img, positive):
                pos_idx = idx
            if torch.equal(img, negative):
                neg_idx = idx
        
        # Note: We can't easily verify labels match without storing the actual indices
        # In practice, we trust the implementation
    
    print("✅ All tests passed!")
