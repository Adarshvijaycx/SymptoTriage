"""Three-state encoder module (+1 present, -1 absent, 0 unknown).

Converts raw symptom text records into a numerical feature matrix
suitable for tree-based classifiers. Implements Informative Missing
Not At Random (I-MNAR) encoding where:
  +1 = symptom confirmed present
  -1 = symptom confirmed absent (only at inference time)
   0 = symptom not recorded / unknown
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple

from sklearn.preprocessing import LabelEncoder


class ThreeStateEncoder:
    """Encodes patient symptom records into a 3-state numeric matrix.

    During training:
      Symptoms listed in the record → +1
      Symptoms NOT listed → 0 (unknown, not queried)

    During inference (when a doctor explicitly marks symptoms):
      Symptom present → +1
      Symptom confirmed absent → -1
      Symptom not asked → 0

    Parameters
    ----------
    symptom_cols : list of str, optional
        Column names containing symptom strings. If None, auto-detected
        from columns named 's1', 's2', etc.
    """

    def __init__(self, symptom_cols: Optional[List[str]] = None):
        self.symptom_cols = symptom_cols
        self.all_symptoms_: Optional[List[str]] = None
        self.symptom_to_idx_: Optional[dict] = None
        self.label_encoder_: Optional[LabelEncoder] = None
        self.n_features_: int = 0

    def fit(self, df: pd.DataFrame, disease_col: str = "disease") -> "ThreeStateEncoder":
        """Learn all unique symptom tokens from the dataset.

        Builds the complete symptom vocabulary that defines the
        feature matrix dimensions.

        Parameters
        ----------
        df : pd.DataFrame
            Canonicalized dataframe with disease label and symptom columns.
        disease_col : str
            Name of the disease label column.

        Returns
        -------
        self
        """
        # Auto-detect symptom columns
        if self.symptom_cols is None:
            self.symptom_cols = [c for c in df.columns if c != disease_col]

        # Discover all unique symptoms
        all_syms = set()
        for col in self.symptom_cols:
            for val in df[col].dropna().unique():
                token = str(val).strip()
                if token:
                    all_syms.add(token)

        self.all_symptoms_ = sorted(all_syms)
        self.symptom_to_idx_ = {s: i for i, s in enumerate(self.all_symptoms_)}
        self.n_features_ = len(self.all_symptoms_)

        # Fit label encoder for disease targets
        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(df[disease_col].values)

        return self

    def transform(self, df: pd.DataFrame,
                  disease_col: str = "disease") -> Tuple[np.ndarray, np.ndarray]:
        """Transform raw records into a binary feature matrix.

        Parameters
        ----------
        df : pd.DataFrame
            Canonicalized dataframe.
        disease_col : str
            Name of the disease label column.

        Returns
        -------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix with +1 (present) and 0 (not recorded).
        y : np.ndarray, shape (n_samples,)
            Integer-encoded disease labels.
        """
        n_samples = len(df)
        X = np.zeros((n_samples, self.n_features_), dtype=np.int8)

        for col in self.symptom_cols:
            if col not in df.columns:
                continue
            for row_i, val in enumerate(df[col]):
                if pd.notna(val):
                    token = str(val).strip()
                    if token in self.symptom_to_idx_:
                        X[row_i, self.symptom_to_idx_[token]] = 1

        y = self.label_encoder_.transform(df[disease_col].values)

        return X, y

    def fit_transform(self, df: pd.DataFrame,
                      disease_col: str = "disease") -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform in one step."""
        return self.fit(df, disease_col).transform(df, disease_col)

    # ── Binary-matrix path (new Diseases_and_Symptoms dataset) ────────────────
    # The new dataset is already a wide 0/1 matrix: each *column* is a symptom
    # and each *cell* is its presence flag. The symptom vocabulary is therefore
    # the column set itself — there are no free-text tokens to discover. These
    # methods populate exactly the same attributes (all_symptoms_,
    # symptom_to_idx_, n_features_, label_encoder_) as the legacy path, so
    # encode_inference_input() and the 3-state (+1/-1/0) inference logic work
    # identically regardless of which dataset the model was trained on.

    def fit_binary_matrix(self, df: pd.DataFrame,
                          disease_col: str = "disease") -> "ThreeStateEncoder":
        """Learn the symptom vocabulary from the matrix's column names."""
        self.symptom_cols = [c for c in df.columns if c != disease_col]
        # Preserve column order as the canonical feature order.
        self.all_symptoms_ = list(self.symptom_cols)
        self.symptom_to_idx_ = {s: i for i, s in enumerate(self.all_symptoms_)}
        self.n_features_ = len(self.all_symptoms_)

        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(df[disease_col].values)
        return self

    def transform_binary_matrix(self, df: pd.DataFrame,
                                disease_col: str = "disease") -> Tuple[np.ndarray, np.ndarray]:
        """Read the already-binary symptom columns straight into a matrix."""
        # Reindex to the learned column order so the feature axis is stable even
        # if the incoming frame's columns are reordered.
        X = df[self.all_symptoms_].to_numpy(dtype=np.int8)
        y = self.label_encoder_.transform(df[disease_col].values)
        return X, y

    def fit_transform_binary_matrix(self, df: pd.DataFrame,
                                    disease_col: str = "disease") -> Tuple[np.ndarray, np.ndarray]:
        """Fit vocabulary from columns and return (X, y) in one step."""
        return self.fit_binary_matrix(df, disease_col).transform_binary_matrix(df, disease_col)

    def encode_inference_input(self, symptoms: dict) -> np.ndarray:
        """Encode a single patient's symptoms for inference.

        Supports the full 3-state encoding:
        - symptoms with value +1  → present
        - symptoms with value -1  → confirmed absent
        - symptoms not in dict    → 0 (unknown)

        Parameters
        ----------
        symptoms : dict
            Mapping of symptom name to value (+1, -1, or 0).
            Example: {"fever": 1, "cough": -1, "headache": 1}

        Returns
        -------
        X : np.ndarray, shape (1, n_features)
            Feature vector for the patient.
        """
        X = np.zeros((1, self.n_features_), dtype=np.float32)

        for symptom_name, value in symptoms.items():
            # Try exact match first
            idx = self.symptom_to_idx_.get(symptom_name)
            if idx is not None:
                X[0, idx] = float(value)
            else:
                # Try case-insensitive match
                lower_name = symptom_name.lower().strip()
                for known, known_idx in self.symptom_to_idx_.items():
                    if known.lower() == lower_name:
                        X[0, known_idx] = float(value)
                        break

        return X

    def get_feature_names(self) -> List[str]:
        """Return ordered list of symptom feature names."""
        return list(self.all_symptoms_) if self.all_symptoms_ else []

    def get_disease_names(self) -> List[str]:
        """Return ordered list of disease class names."""
        if self.label_encoder_ is not None:
            return list(self.label_encoder_.classes_)
        return []

    def decode_label(self, label_idx: int) -> str:
        """Convert integer label back to disease name."""
        return self.label_encoder_.inverse_transform([label_idx])[0]

    def decode_labels(self, label_indices: np.ndarray) -> np.ndarray:
        """Convert array of integer labels back to disease names."""
        return self.label_encoder_.inverse_transform(label_indices)
