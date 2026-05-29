"""FastAPI app entry point."""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import PredictRequest, PredictResponse
from src.api.predict import ModelService
from src.api.model_loader import ensure_models


service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Fetch large artifacts (if needed) then load model artifacts on startup."""
    global service
    models_dir = os.environ.get("MODELS_DIR", "models")
    try:
        # Pull any large artifacts kept out of the image from object storage.
        # No-op when files already exist locally with valid checksums.
        fetched = ensure_models(models_dir)
        if fetched:
            print(f"Fetched model artifacts: {', '.join(fetched)}")
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
# `allow_origins=["*"]` together with `allow_credentials=True` is invalid:
# browsers refuse to honor a wildcard origin on credentialed requests, and a
# wildcard on an unauthenticated medical endpoint is an unsafe default. The
# frontend uses plain fetch (no cookies/auth), so credentials are not needed.
# Origins come from CORS_ALLOW_ORIGINS (comma-separated); default to localhost.
_default_origins = "http://127.0.0.1:8000,http://localhost:8000,http://127.0.0.1:5500,http://localhost:5500"
allowed_origins = [
    o.strip()
    for o in os.environ.get("CORS_ALLOW_ORIGINS", _default_origins).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
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


@app.get("/drift")
def drift_status() -> dict:
    """Report PSI drift over buffered inference requests vs. training data."""
    if not service:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return service.check_drift()


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
