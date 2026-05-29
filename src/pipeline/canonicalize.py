"""Canonicalization utilities for symptom name normalization.

Handles synonym mapping, whitespace cleanup, and disease label
standardization from the raw dataset.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


# ── Synonym map: alternate names → canonical form ─────────────────────────────
# Keys are lowercase alternates, values are the canonical symptom name.
SYNONYM_MAP: Dict[str, str] = {
    # Whitespace / formatting variants found in dataset
    "dischromic _patches": "dischromic_patches",
    "spotting_ urination": "spotting_urination",
    "foul_smell_of urine": "foul_smell_of_urine",
    # Common clinical synonyms a doctor might use
    "pyrexia": "high_fever",
    "high temp": "high_fever",
    "temperature": "high_fever",
    "febrile": "high_fever",
    "sob": "breathlessness",
    "dyspnoea": "breathlessness",
    "shortness of breath": "breathlessness",
    "coughing": "cough",
    "productive cough": "cough",
    "dry cough": "cough",
    "emesis": "vomiting",
    "throwing up": "vomiting",
    "loose stools": "diarrhoea",
    "diarrhea": "diarrhoea",
    "tummy pain": "stomach_pain",
    "abdominal cramps": "abdominal_pain",
    "weight reduction": "weight_loss",
    "tiredness": "fatigue",
    "exhaustion": "fatigue",
    "pruritis": "itching",
    "rash": "skin_rash",
    "urticaria": "skin_rash",
    "vertigo": "dizziness",
    "migraine headache": "headache",
    "cephalgia": "headache",
    "perspiration": "sweating",
    "diaphoresis": "sweating",
    "anorexia": "loss_of_appetite",
    "poor appetite": "loss_of_appetite",
    "jaundiced eyes": "yellowing_of_eyes",
    "icteric eyes": "yellowing_of_eyes",
    "jaundiced skin": "yellowish_skin",
    "icteric skin": "yellowish_skin",
    "tachycardia": "fast_heart_rate",
    "rapid heartbeat": "fast_heart_rate",
}

# ── Disease label normalization ───────────────────────────────────────────────
# Maps raw disease labels to clean, standardized versions
DISEASE_LABEL_MAP: Dict[str, str] = {
    "Diabetes ": "Diabetes",
    "Hypertension ": "Hypertension",
    "hepatitis A": "Hepatitis A",
    "(vertigo) Paroymsal  Positional Vertigo": "Vertigo (BPPV)",
    "Dimorphic hemmorhoids(piles)": "Hemorrhoids (Piles)",
    "Peptic ulcer diseae": "Peptic Ulcer Disease",
    "Osteoarthristis": "Osteoarthritis",
}


class SymptomCanonicalizer:
    """Normalizes symptom names and disease labels from raw CSV data.

    Handles:
    - Leading/trailing whitespace in symptom tokens
    - Synonym mapping to canonical names
    - Case normalization
    - Disease label standardization
    """

    def __init__(self, synonym_map: Optional[Dict[str, str]] = None,
                 disease_map: Optional[Dict[str, str]] = None):
        self.synonym_map = synonym_map or SYNONYM_MAP
        self.disease_map = disease_map or DISEASE_LABEL_MAP
        # Lowercase keys for case-insensitive matching
        self._syn_lower = {k.lower(): v for k, v in self.synonym_map.items()}
        self.known_symptoms_: Optional[List[str]] = None

    def fit(self, df: pd.DataFrame) -> "SymptomCanonicalizer":
        """Learn all unique symptom tokens from the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Raw dataframe where column 0 is disease label and
            columns 1..N are symptom strings (may contain NaN).

        Returns
        -------
        self
        """
        symptom_cols = df.columns[1:]
        raw_symptoms = set()
        for col in symptom_cols:
            for val in df[col].dropna().unique():
                canon = self._canonicalize_token(val)
                if canon:
                    raw_symptoms.add(canon)
        self.known_symptoms_ = sorted(raw_symptoms)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply canonicalization to all symptom columns and disease labels.

        Parameters
        ----------
        df : pd.DataFrame
            Raw dataframe with disease label in column 0 and symptoms in
            remaining columns.

        Returns
        -------
        pd.DataFrame
            Cleaned dataframe with normalized symptom tokens and disease
            labels.
        """
        result = df.copy()

        # Normalize disease labels
        result.iloc[:, 0] = result.iloc[:, 0].apply(self._clean_disease)

        # Normalize symptom columns
        for col in result.columns[1:]:
            result[col] = result[col].apply(
                lambda x: self._canonicalize_token(x) if pd.notna(x) else np.nan
            )

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def _canonicalize_token(self, raw: str) -> Optional[str]:
        """Normalize a single symptom token.

        Steps:
        1. Strip whitespace
        2. Check synonym map (case-insensitive)
        3. Return canonical form or cleaned original
        """
        if not isinstance(raw, str):
            return None
        cleaned = raw.strip()
        if not cleaned:
            return None

        # Check synonym map (case-insensitive)
        lower = cleaned.lower()
        if lower in self._syn_lower:
            return self._syn_lower[lower]

        # Check original form in synonym map (handles mixed-case keys)
        if cleaned in self.synonym_map:
            return self.synonym_map[cleaned]

        return cleaned

    def _clean_disease(self, raw) -> str:
        """Normalize a disease label."""
        if pd.isna(raw):
            return raw
        cleaned = str(raw).strip()
        return self.disease_map.get(cleaned, cleaned)


def load_raw_dataset(csv_path: str) -> pd.DataFrame:
    """Load the raw symptom-disease CSV file (LEGACY name-column format).

    The CSV has no header row. Column 0 is the disease label,
    columns 1-17 are symptom strings (variable length, padded with
    empty strings / NaN).

    Parameters
    ----------
    csv_path : str
        Path to dataset.csv

    Returns
    -------
    pd.DataFrame
        Raw dataframe with columns ['disease', 's1', 's2', ..., 's17']
    """
    df = pd.read_csv(csv_path, header=None)

    # First row might be empty headers or explicit headers — check
    first_cell = str(df.iloc[0, 0]).strip()
    if pd.isna(df.iloc[0, 0]) or first_cell == "" or first_cell.lower() == "disease":
        df = df.iloc[1:].reset_index(drop=True)

    # Name columns
    n_sym_cols = df.shape[1] - 1
    df.columns = ["disease"] + [f"s{i}" for i in range(1, n_sym_cols + 1)]

    # Drop rows with no disease label
    df = df.dropna(subset=["disease"]).reset_index(drop=True)
    df["disease"] = df["disease"].str.strip()

    # Replace empty strings with NaN
    df = df.replace("", np.nan)

    return df


# ── New (binary-matrix) dataset support ───────────────────────────────────────
# The Diseases_and_Symptoms dataset is already a wide 0/1 matrix: the first
# column is the disease label and every other column is a symptom flag. There
# are no free-text symptom tokens to normalize, so canonicalization is a no-op
# here (the columns themselves ARE the canonical symptom vocabulary).

def load_binary_matrix_dataset(csv_path: str,
                               label_col: str = "diseases") -> pd.DataFrame:
    """Load a wide binary symptom matrix (new dataset format).

    Expects a headered CSV where ``label_col`` holds the disease name and all
    remaining columns are 0/1 symptom indicators. The label column is renamed
    to ``"disease"`` so it matches the downstream ``disease_col`` convention
    used by ThreeStateEncoder / FeatureEngineer.

    Parameters
    ----------
    csv_path : str
        Path to the binary-matrix CSV.
    label_col : str, default="diseases"
        Name of the disease label column in the source file.

    Returns
    -------
    pd.DataFrame
        Dataframe with column ``"disease"`` first, followed by integer 0/1
        symptom columns (column names preserved as the symptom vocabulary).
    """
    df = pd.read_csv(csv_path)

    if label_col not in df.columns:
        raise ValueError(
            f"label_col '{label_col}' not found in {csv_path}. "
            f"First columns are: {list(df.columns[:5])}"
        )

    # Standardize the label column name for downstream consistency.
    df = df.rename(columns={label_col: "disease"})

    # Move the disease column to the front if it isn't already.
    cols = ["disease"] + [c for c in df.columns if c != "disease"]
    df = df[cols]

    # Clean labels and drop unlabeled rows.
    df = df.dropna(subset=["disease"]).reset_index(drop=True)
    df["disease"] = df["disease"].astype(str).str.strip()

    # Coerce symptom columns to a compact integer 0/1 form. Any stray NaN in a
    # symptom cell means "not present" -> 0.
    sym_cols = [c for c in df.columns if c != "disease"]
    df[sym_cols] = df[sym_cols].fillna(0).astype(np.int8)

    return df


class PassThroughCanonicalizer:
    """No-op canonicalizer for the already-clean binary-matrix dataset.

    Mirrors the SymptomCanonicalizer interface (fit / transform /
    fit_transform) so the training pipeline can treat both dataset formats
    uniformly. Symptom columns are pre-encoded 0/1 flags, so the only thing
    worth normalizing is stray whitespace on the disease label.
    """

    def __init__(self):
        self.known_symptoms_: Optional[List[str]] = None

    def fit(self, df: pd.DataFrame) -> "PassThroughCanonicalizer":
        self.known_symptoms_ = [c for c in df.columns if c != "disease"]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        if "disease" in result.columns:
            result["disease"] = result["disease"].astype(str).str.strip()
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
