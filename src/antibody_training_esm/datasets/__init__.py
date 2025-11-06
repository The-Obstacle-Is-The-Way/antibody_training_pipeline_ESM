"""
Dataset Loaders Module

Professional dataset loaders following Open/Closed Principle.

IMPORTANT: These classes LOAD preprocessed data, they do NOT run preprocessing pipelines.
For preprocessing, use the scripts in: preprocessing/<dataset>/

This module provides:
- AntibodyDataset: Abstract base class for all dataset loaders
- JainDataset: Jain 2017 therapeutic antibody dataset loader
- HarveyDataset: Harvey nanobody polyreactivity dataset loader
- ShehataDataset: Shehata HIV antibody dataset loader
- BoughterDataset: Boughter mouse antibody dataset loader

Each dataset class:
1. Implements dataset-specific loading logic for PREPROCESSED data
2. Provides common utilities (validation, statistics)
3. Can be extended without modifying existing code (OCP)

Example usage:
    >>> from antibody_training_esm.datasets import HarveyDataset
    >>> dataset = HarveyDataset()
    >>> df = dataset.load_data()
    >>> print(f"Loaded {len(df)} sequences")
"""

from .base import AntibodyDataset
from .boughter import BoughterDataset, load_boughter_data
from .harvey import HarveyDataset, load_harvey_data
from .jain import JainDataset, load_jain_data
from .shehata import ShehataDataset, load_shehata_data

__all__ = [
    # Base class
    "AntibodyDataset",
    # Concrete dataset loader classes
    "JainDataset",
    "HarveyDataset",
    "ShehataDataset",
    "BoughterDataset",
    # Convenience functions for loading
    "load_jain_data",
    "load_harvey_data",
    "load_shehata_data",
    "load_boughter_data",
]
