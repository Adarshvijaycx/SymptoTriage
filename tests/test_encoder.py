import pytest
import pandas as pd
import numpy as np

from src.pipeline.canonicalize import SymptomCanonicalizer
from src.pipeline.encoder import ThreeStateEncoder


def test_canonicalizer():
    df = pd.DataFrame({
        "disease": [" Diabetes ", "Acne"],
        "sym1": ["  high temp  ", "spotting_ urination"],
        "sym2": ["sob", None]
    })
    
    canon = SymptomCanonicalizer()
    df_clean = canon.fit_transform(df)
    
    # Check diseases cleaned
    assert df_clean.iloc[0, 0] == "Diabetes"
    
    # Check synonyms mapped
    assert df_clean.iloc[0, 1] == "high_fever"
    assert df_clean.iloc[0, 2] == "breathlessness"
    
    # Check whitespace handled
    assert df_clean.iloc[1, 1] == "spotting_urination"


def test_three_state_encoder():
    df = pd.DataFrame({
        "disease": ["flu", "flu", "cold"],
        "s1": ["fever", "cough", "headache"],
        "s2": ["cough", None, "fever"]
    })
    
    encoder = ThreeStateEncoder()
    X, y = encoder.fit_transform(df)
    
    # Features should be alphabetically sorted: cough, fever, headache
    feats = encoder.get_feature_names()
    assert feats == ["cough", "fever", "headache"]
    
    # flu row 1 (fever, cough) -> 1, 1, 0
    assert list(X[0]) == [1, 1, 0]
    
    # flu row 2 (cough, None) -> 1, 0, 0
    assert list(X[1]) == [1, 0, 0]
    
    # encode_inference_input test
    X_infer = encoder.encode_inference_input({
        "fever": 1,
        "cough": -1,
        "headache": 0
    })
    
    assert list(X_infer[0]) == [-1.0, 1.0, 0.0]
