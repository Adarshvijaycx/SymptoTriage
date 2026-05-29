"""Feature engineering helpers for interaction and temporal features.

Adds derived features on top of the raw binary symptom matrix to
improve model discrimination:
  - symptom_count: total number of symptoms present per patient
  - category_symptom_counts: count of symptoms per organ-system group
  - pairwise interactions (optional): selected high-value pairs
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


# ── Symptom category groupings ────────────────────────────────────────────────
# Used to compute per-category symptom counts as features
SYMPTOM_CATEGORIES: Dict[str, List[str]] = {
    "respiratory": [
        "cough", "breathlessness", "phlegm", "mucoid_sputum",
        "rusty_sputum", "blood_in_sputum", "chest_pain",
        "congestion", "runny_nose", "sinus_pressure",
        "throat_irritation", "continuous_sneezing", "loss_of_smell",
    ],
    "gastrointestinal": [
        "vomiting", "nausea", "diarrhoea", "constipation",
        "stomach_pain", "abdominal_pain", "belly_pain",
        "indigestion", "acidity", "loss_of_appetite",
        "passage_of_gases", "internal_itching",
        "ulcers_on_tongue", "stomach_bleeding",
    ],
    "dermatological": [
        "skin_rash", "itching", "nodal_skin_eruptions",
        "dischromic_patches", "skin_peeling", "pus_filled_pimples",
        "blackheads", "scurring", "blister",
        "red_sore_around_nose", "yellow_crust_ooze",
        "red_spots_over_body", "silver_like_dusting",
        "small_dents_in_nails", "inflammatory_nails",
    ],
    "musculoskeletal": [
        "joint_pain", "muscle_pain", "back_pain", "neck_pain",
        "knee_pain", "hip_joint_pain", "muscle_weakness",
        "muscle_wasting", "weakness_in_limbs", "swelling_joints",
        "movement_stiffness", "stiff_neck", "painful_walking",
        "cramps",
    ],
    "neurological": [
        "headache", "dizziness", "loss_of_balance",
        "altered_sensorium", "weakness_of_one_body_side",
        "spinning_movements", "unsteadiness", "visual_disturbances",
        "blurred_and_distorted_vision", "lack_of_concentration",
        "depression", "irritability", "anxiety", "slurred_speech",
    ],
    "systemic": [
        "fatigue", "lethargy", "restlessness", "malaise",
        "high_fever", "mild_fever", "chills", "sweating",
        "shivering", "weight_loss", "weight_gain", "obesity",
        "dehydration",
    ],
    "hepatic": [
        "yellowish_skin", "yellowing_of_eyes", "dark_urine",
        "yellow_urine", "acute_liver_failure",
        "history_of_alcohol_consumption", "fluid_overload",
        "swelling_of_stomach", "distention_of_abdomen",
    ],
    "metabolic_endocrine": [
        "excessive_hunger", "increased_appetite", "polyuria",
        "irregular_sugar_level", "mood_swings",
        "cold_hands_and_feets", "enlarged_thyroid",
        "puffy_face_and_eyes", "brittle_nails",
        "swollen_extremeties", "abnormal_menstruation",
        "fast_heart_rate", "palpitations",
        "drying_and_tingling_lips",
    ],
}


# ── Binary-matrix symptom groupings (new Diseases_and_Symptoms dataset) ───────
# The new dataset's columns are natural-language symptom names, so the legacy
# SYMPTOM_CATEGORIES above (old token names) would match nothing and produce
# all-zero category-count features. These groupings are built from the actual
# 230 columns and cover every symptom column exactly once.
BINARY_MATRIX_SYMPTOM_CATEGORIES: Dict[str, List[str]] = {
      "respiratory": [
        "shortness of breath", "breathing fast", "cough",
        "nasal congestion", "wheezing", "difficulty breathing",
        "coughing up sputum", "hemoptysis", "congestion in chest",
        "apnea", "abnormal breathing sounds", "sinus congestion",
        "sneezing", "coryza", "hurts to breath",
        "throat swelling", "sore throat", "hoarse voice",
      ],
      "cardiac": [
        "sharp chest pain", "chest tightness", "palpitations",
        "irregular heartbeat", "decreased heart rate", "increased heart rate",
        "burning chest pain", "peripheral edema",
      ],
      "gastrointestinal": [
        "sharp abdominal pain", "vomiting", "nausea",
        "diarrhea", "lower abdominal pain", "vomiting blood",
        "regurgitation", "burning abdominal pain", "heartburn",
        "blood in stool", "upper abdominal pain", "stomach bloating",
        "changes in stool appearance", "melena", "rectal bleeding",
        "constipation", "difficulty in swallowing", "regurgitation.1",
        "side pain", "jaundice", "mass or swelling around the anus",
        "pain of the anus", "itching of the anus",
      ],
      "genitourinary": [
        "retention of urine", "painful urination", "involuntary urination",
        "frequent urination", "blood in urine", "unusual color or odor to urine",
        "excessive urination at night", "kidney mass", "symptoms of the kidneys",
        "symptoms of prostate", "symptoms of bladder", "suprapubic pain",
        "hesitancy", "low urine output", "impotence",
        "symptoms of the scrotum and testes", "swelling of scrotum", "pain in testicles",
      ],
      "gynecologic_pregnancy": [
        "vaginal itching", "pain during intercourse", "vaginal discharge",
        "intermenstrual bleeding", "vaginal pain", "vaginal redness",
        "pain during pregnancy", "pelvic pain", "spotting or bleeding during pregnancy",
        "problems during pregnancy", "blood clots during menstrual periods", "long menstrual periods",
        "heavy menstrual flow", "unpredictable menstruation", "painful menstruation",
        "frequent menstruation", "recent pregnancy", "uterine contractions",
        "infertility", "hot flashes",
      ],
      "musculoskeletal": [
        "leg pain", "hip pain", "hand or finger pain",
        "wrist pain", "hand or finger swelling", "arm pain",
        "wrist swelling", "arm stiffness or tightness", "arm swelling",
        "hand or finger stiffness or tightness", "back pain", "neck pain",
        "low back pain", "knee pain", "foot or toe pain",
        "ankle pain", "bones are painful", "elbow pain",
        "knee swelling", "problems with movement", "knee stiffness or tightness",
        "leg swelling", "foot or toe swelling", "shoulder pain",
        "shoulder stiffness or tightness", "ache all over", "lower body pain",
        "rib pain", "joint pain", "cramps and spasms",
        "back cramps or spasms", "back stiffness or tightness", "hip stiffness or tightness",
        "ankle swelling", "elbow swelling", "arm weakness",
        "leg weakness", "hand or finger weakness", "arm lump or mass",
        "back mass or lump", "hand or finger lump or mass",
      ],
      "neurological": [
        "dizziness", "abnormal involuntary movements", "difficulty speaking",
        "fainting", "headache", "frontal headache",
        "loss of sensation", "focal weakness", "disturbance of memory",
        "paresthesia", "seizures", "weakness",
        "abnormal movement of eyelid",
      ],
      "psychiatric": [
        "anxiety and nervousness", "depression", "depressive or psychotic symptoms",
        "insomnia", "abusing alcohol", "hostile behavior",
        "drug abuse", "restlessness", "excessive anger",
        "delusions or hallucinations", "temper problems", "fears and phobias",
        "low self-esteem", "obsessions and compulsions", "antisocial behavior",
        "hysterical behavior", "sleepiness",
      ],
      "dermatological": [
        "skin swelling", "abnormal appearing skin", "skin lesion",
        "acne or pimples", "skin growth", "irregular appearing scalp",
        "skin moles", "irregular appearing nails", "itching of skin",
        "skin dryness, peeling, scaliness, or roughness", "skin irritation", "itchy scalp",
        "warts", "skin rash", "lip swelling",
        "allergic reaction",
      ],
      "ophthalmic": [
        "white discharge from eye", "diminished vision", "double vision",
        "symptoms of eye", "pain in eye", "foreign body sensation in eye",
        "spots or clouds in vision", "eye redness", "lacrimation",
        "itchiness of eye", "blindness", "eye burns or stings",
        "bleeding from eye", "mass on eyelid", "swollen eye",
        "eyelid swelling", "eyelid lesion or rash",
      ],
      "ent_oral": [
        "diminished hearing", "pus draining from ear", "ear pain",
        "ringing in ear", "plugged feeling in ear", "itchy ear(s)",
        "fluid in ear", "pulling at ears", "redness in ear",
        "bleeding from ear", "toothache", "mouth ulcer",
        "mouth dryness", "jaw swelling", "neck mass",
        "neck swelling", "gum pain", "nosebleed",
        "painful sinuses", "bleeding gums", "pain in gums",
        "mouth pain", "swollen or red tonsils", "facial pain",
        "symptoms of the face",
      ],
      "constitutional": [
        "feeling ill", "fever", "chills",
        "fatigue", "sweating", "weight gain",
        "decreased appetite", "fluid retention", "flu-like syndrome",
        "groin pain", "lack of growth", "irritable infant",
        "infant feeding problem", "diaper rash",
      ],
}

# Legacy interaction pairs (old dataset token names) — default for back-compat.
LEGACY_INTERACTION_PAIRS: List = [
    ("high_fever", "chills"),
    ("yellowish_skin", "dark_urine"),
    ("joint_pain", "muscle_pain"),
    ("breathlessness", "chest_pain"),
    ("itching", "skin_rash"),
    ("fatigue", "weight_loss"),
    ("nausea", "vomiting"),
    ("headache", "dizziness"),
]

# Clinically-motivated interaction pairs for the new binary-matrix columns.
BINARY_MATRIX_INTERACTION_PAIRS: List = [
    ("fever", "chills"),
    ("cough", "shortness of breath"),
    ("nausea", "vomiting"),
    ("headache", "dizziness"),
    ("sharp abdominal pain", "vomiting"),
    ("skin rash", "itching of skin"),
    ("painful urination", "frequent urination"),
    ("anxiety and nervousness", "depression"),
    ("sharp chest pain", "shortness of breath"),
    ("joint pain", "ache all over"),
    ("sore throat", "swollen or red tonsils"),
    ("nasal congestion", "sneezing"),
]


class FeatureEngineer:
    """Adds derived features to the binary symptom matrix.

    Parameters
    ----------
    feature_names : list of str
        Ordered list of symptom feature names (from ThreeStateEncoder).
    symptom_categories : dict, optional
        Mapping of category name → list of symptom names.
    add_interactions : bool
        Whether to add pairwise interaction features.
    """

    def __init__(self, feature_names: List[str],
                 symptom_categories: Optional[Dict[str, List[str]]] = None,
                 add_interactions: bool = False,
                 interaction_pairs: Optional[List] = None):
        self.feature_names = feature_names
        self.symptom_categories = symptom_categories or SYMPTOM_CATEGORIES
        self.add_interactions = add_interactions
        # Pairs default to the legacy old-dataset tokens for back-compat; the
        # binary-matrix path passes BINARY_MATRIX_INTERACTION_PAIRS explicitly.
        self.interaction_pairs = (interaction_pairs if interaction_pairs is not None
                                  else LEGACY_INTERACTION_PAIRS)

        # Build category index maps
        self._cat_indices: Dict[str, List[int]] = {}
        feat_set = {f: i for i, f in enumerate(feature_names)}
        for cat_name, symptoms in self.symptom_categories.items():
            indices = [feat_set[s] for s in symptoms if s in feat_set]
            if indices:
                self._cat_indices[cat_name] = indices

        self.derived_feature_names_: Optional[List[str]] = None

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Add derived features to the feature matrix.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_base_features)
            Binary symptom matrix from ThreeStateEncoder.

        Returns
        -------
        X_enriched : np.ndarray, shape (n_samples, n_base_features + n_derived)
            Matrix with appended derived features.
        """
        derived_cols = []
        derived_names = []

        # 1. Total symptom count
        symptom_count = (X > 0).sum(axis=1, keepdims=True).astype(np.float32)
        derived_cols.append(symptom_count)
        derived_names.append("symptom_count")

        # 2. Per-category symptom counts
        for cat_name, indices in self._cat_indices.items():
            cat_count = (X[:, indices] > 0).sum(axis=1, keepdims=True).astype(np.float32)
            derived_cols.append(cat_count)
            derived_names.append(f"cat_{cat_name}_count")

        # 3. Pairwise interactions (optional — expensive for wide matrices)
        if self.add_interactions:
            # Pairs come from self.interaction_pairs (legacy tokens by default,
            # or binary-matrix column names when configured for the new dataset).
            feat_idx = {f: i for i, f in enumerate(self.feature_names)}
            for s1, s2 in self.interaction_pairs:
                if s1 in feat_idx and s2 in feat_idx:
                    interaction = (X[:, feat_idx[s1]] * X[:, feat_idx[s2]]).reshape(-1, 1)
                    derived_cols.append(interaction.astype(np.float32))
                    derived_names.append(f"interaction_{s1}_x_{s2}")

        self.derived_feature_names_ = derived_names

        if derived_cols:
            X_derived = np.hstack(derived_cols)
            return np.hstack([X.astype(np.float32), X_derived])
        return X.astype(np.float32)

    def get_all_feature_names(self) -> List[str]:
        """Return complete feature name list (base + derived)."""
        names = list(self.feature_names)
        if self.derived_feature_names_:
            names.extend(self.derived_feature_names_)
        return names
