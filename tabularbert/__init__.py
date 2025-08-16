"""
TabularBERT Package

A comprehensive framework for tabular data modeling using BERT-based transformers.
Provides self-supervised pretraining and supervised fine-tuning capabilities.
"""

from .model import TabularBERT, TabularBERTTrainer
from .utils import QuantileDiscretize

__version__ = "0.1.0"
__author__ = "Beomjin Park"
__email__ = "bbeomjin@gmail.com"

__all__ = [
    # Core models
    'TabularBERT',
    'TabularBERTTrainer', 
    
    # Data utilities
    'QuantileDiscretize',
]
