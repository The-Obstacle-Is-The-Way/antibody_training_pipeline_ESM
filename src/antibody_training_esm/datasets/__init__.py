"""
Dataset Abstractions Module

Professional dataset preprocessors following Open/Closed Principle.

This module provides:
- AntibodyDataset: Abstract base class for all dataset preprocessors
- JainDataset: Jain 2017 therapeutic antibody dataset
- HarveyDataset: Harvey nanobody polyreactivity dataset
- ShehataDataset: Shehata HIV antibody dataset
- BoughterDataset: Boughter mouse antibody dataset

Each dataset class:
1. Implements dataset-specific loading logic
2. Shares common preprocessing operations (annotation, fragment generation)
3. Can be extended without modifying existing code (OCP)

Example usage:
    >>> from antibody_training_esm.datasets import HarveyDataset
    >>> dataset = HarveyDataset()
    >>> df = dataset.process()
    >>> print(f"Processed {len(df)} sequences")
"""

from .base import AntibodyDataset
from .boughter import BoughterDataset, preprocess_boughter
from .harvey import HarveyDataset, preprocess_harvey
from .jain import JainDataset, preprocess_jain
from .shehata import ShehataDataset, preprocess_shehata

__all__ = [
    # Base class
    "AntibodyDataset",
    # Concrete dataset classes
    "JainDataset",
    "HarveyDataset",
    "ShehataDataset",
    "BoughterDataset",
    # Convenience functions
    "preprocess_jain",
    "preprocess_harvey",
    "preprocess_shehata",
    "preprocess_boughter",
]
