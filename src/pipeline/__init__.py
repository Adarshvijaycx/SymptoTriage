"""Pipeline module — data loading, canonicalization, encoding, feature engineering."""

from src.pipeline.canonicalize import (
    SymptomCanonicalizer,
    load_raw_dataset,
    SYNONYM_MAP,
    DISEASE_LABEL_MAP,
)
from src.pipeline.encoder import ThreeStateEncoder
from src.pipeline.feature_eng import FeatureEngineer

__all__ = [
    "SymptomCanonicalizer",
    "load_raw_dataset",
    "SYNONYM_MAP",
    "DISEASE_LABEL_MAP",
    "ThreeStateEncoder",
    "FeatureEngineer",
]
