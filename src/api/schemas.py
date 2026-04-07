"""Pydantic request/response models for prediction API."""

from pydantic import BaseModel, Field
from typing import Dict, Optional


class PredictRequest(BaseModel):
    """Payload for patient diagnostic prediction."""
    symptoms: Dict[str, float] = Field(
        ...,
        description="Dictionary mapping symptom names to 1 (present), -1 (absent), or 0 (unknown).",
        example={"high_fever": 1.0, "cough": -1.0, "headache": 1.0}
    )


class PredictResponse(BaseModel):
    """Response payload with diagnosis, confidence, and explanations."""
    disease: str = Field(..., description="Predicted diagnosis string.")
    category: str = Field(..., description="Broad disease category (e.g. Respiratory).")
    probability: float = Field(..., description="Calibrated confidence score (0.0 to 1.0).")
    decision_path: str = Field(..., description="Human-readable decision rule from the Stage-1 router.")
    shap_values: Dict[str, float] = Field(
        ..., 
        description="Top 15 most impactful symptoms and their SHAP values."
    )
    disease_description: str = Field(..., description="Description of the predicted disease.")
    disease_precautions: list[str] = Field(..., description="List of precautions for the predicted disease.")
    symptom_severities: Dict[str, int] = Field(..., description="Severities for the patient's active symptoms.")
    latency_ms: float = Field(..., description="Server-side inference latency.")
