"""SMOTE-NC based augmentation helpers.

SMOTE-NC handles synthetic generation of mixed categorical and
continuous data for minority classes.
"""

from typing import Tuple
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC


class SMOTEHandler:
    """Wrapper for SMOTE-NC augmentation.

    Parameters
    ----------
    categorical_features : list of int
        Indices of categorical features in the feature matrix.
    k_neighbors : int, default=5
        Number of nearest neighbours to used to construct synthetic samples.
    sampling_strategy : str or dict, default='auto'
        Strategy to resample the dataset.
    random_state : int, default=42
        Seed for reproducibility.
    """

    def __init__(self, categorical_features: list[int],
                 k_neighbors: int = 5,
                 sampling_strategy: str | dict = "auto",
                 random_state: int = 42):
        self.categorical_features = categorical_features
        self.model = SMOTENC(
            categorical_features=categorical_features,
            k_neighbors=k_neighbors,
            sampling_strategy=sampling_strategy,
            random_state=random_state
        )

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE-NC to generate synthetic samples.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels

        Returns
        -------
        X_resampled : np.ndarray
        y_resampled : np.ndarray
        """
        # SMOTENC requires feature matrix to be dense and typically works
        # well with pandas DataFrames or pure numpy matrices, but requires
        # the categorical mask explicitly.
        X_res, y_res = self.model.fit_resample(X, y)
        return X_res, y_res
