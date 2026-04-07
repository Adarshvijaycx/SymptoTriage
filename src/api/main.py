"""FastAPI app entry point."""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import PredictRequest, PredictResponse
from src.api.predict import ModelService


service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup."""
    global service
    models_dir = os.environ.get("MODELS_DIR", "models")
    try:
        service = ModelService(models_dir=models_dir)
        print("ModelService online.")
    except Exception as e:
        print(f"Warning: Models failed to load. {e}")
    yield
    # Cleanup on shutdown
    service = None


app = FastAPI(
    title="Medical Diagnosis API",
    version="1.0",
    description="2-Stage Hierarchical Classification with TreeSHAP Explanations",
    lifespan=lifespan
)

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    """Basic health check endpoint."""
    return {
        "status": "ok",
        "models_loaded": service is not None
    }


@app.get("/symptoms")
def list_symptoms() -> dict:
    """Return array of valid symptom names."""
    if not service:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"symptoms": service.get_valid_symptoms()}


@app.get("/diseases")
def list_diseases() -> dict:
    """Return array of valid target disease names."""
    if not service:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"diseases": service.get_valid_diseases()}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    """
    Run diagnostic prediction over a patient's symptom set.
    """
    if not service:
        raise HTTPException(status_code=503, detail="Models not loaded")
        
    try:
        return service.run_prediction(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
