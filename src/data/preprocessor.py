"""
Data preprocessing: scaling, encoding, sequence creation.
"""
import logging
import numpy as np
from typing import Tuple
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pandas as pd
from ..utils.logger import setup_logger
from .schemas import validate_dataframe
from .dataset import KDD99Dataset

logger = setup_logger(__name__)

class KDD99Preprocessor:
    """Preprocessor for KDD99 features."""
    CATEGORICAL_FEATURES
