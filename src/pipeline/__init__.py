"""Pipeline module — data loading, canonicalization, encoding, feature engineering."""

from src.pipeline.canonicalize import (
    SymptomCanonicalizer,
    PassThroughCanonicalizer,
    load_raw_dataset,
    load_binary_matrix_dataset,
    SYNONYM_MAP,
    DISEASE_LABEL_MAP,
)
from src.pipeline.encoder import ThreeStateEncoder
from src.pipeline.feature_eng import FeatureEngineer

__all__ = [
    "SymptomCanonicalizer",
    "PassThroughCanonicalizer",
    "load_raw_dataset",
    "load_binary_matrix_dataset",
    "SYNONYM_MAP",
    "DISEASE_LABEL_MAP",
    "ThreeStateEncoder",
    "FeatureEngineer",
]
