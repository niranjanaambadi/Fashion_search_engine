"""
Models package for fashion search engine.

This package contains all neural network architectures:
- InvertedResidualBlock: Core building block
- MobileNetBackbone: Feature extractor
- MobileNetLikeClassifier: Classification model
- SiameseEncoder: Embedding generator
- SiameseNetwork: Metric learning network
"""

from .blocks import InvertedResidualBlock
from .backbone import MobileNetBackbone
from .classifier import MobileNetLikeClassifier
from .siamese import SiameseEncoder, SiameseNetwork

__all__ = [
    'InvertedResidualBlock',
    'MobileNetBackbone',
    'MobileNetLikeClassifier',
    'SiameseEncoder',
    'SiameseNetwork',
]

__version__ = '0.1.0'
