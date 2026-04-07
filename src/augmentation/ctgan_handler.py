"""CTGAN training and sampling helpers.

Uses a standard CTGAN model to learn the distribution of majority
classes and generate realistic synthetic records.
"""

from typing import List, Optional
import numpy as np
import pandas as pd

try:
    from ctgan import CTGAN
    CTGAN_AVAILABLE = True
except ImportError:
    CTGAN_AVAILABLE = False


class CTGANHandler:
    """Wrapper for CTGAN synthetic data generation.

    Parameters
    ----------
    discrete_columns : list of str (or int indices)
        List of categorical column names/indices to tell CTGAN which
        features are discrete.
    epochs : int, default=100
        Number of training epochs.
    batch_size : int, default=500
        Batch size.
    """

    def __init__(self, discrete_columns: List[int | str],
                 epochs: int = 100,
                 batch_size: int = 500):
        if not CTGAN_AVAILABLE:
            raise ImportError("ctgan package is required. Install via pip install ctgan")
        self.discrete_columns = discrete_columns
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = CTGAN(epochs=epochs, batch_size=batch_size, verbose=False)

    def fit(self, X: pd.DataFrame) -> "CTGANHandler":
        """Train the CTGAN on a given subset of data (usually majority class).

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix as dataframe.

        Returns
        -------
        self
        """
        self.model.fit(X, self.discrete_columns)
        return self

    def sample(self, n_samples: int) -> pd.DataFrame:
        """Sample synthetic records from the trained CTGAN.

        Parameters
        ----------
        n_samples : int
            Number of requested samples.

        Returns
        -------
        pd.DataFrame
            Synthetic records.
        """
        return self.model.sample(n_samples)
