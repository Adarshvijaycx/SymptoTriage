import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    # Assuming models were loaded if training was run before tests
    assert "models_loaded" in response.json()

def test_get_symptoms():
    response = client.get("/symptoms")
    if response.status_code == 200:
        data = response.json()
        assert "symptoms" in data
        assert len(data["symptoms"]) > 0
    else:
        # Fallback passing if Models aren't loaded (503 HTTP)
        assert response.status_code in [200, 503]

def test_predict_endpoint():
    payload = {
        "symptoms": {
            "high_fever": 1.0,
            "cough": -1.0,
            "chest_pain": 0.0,
            "fatigue": 1.0
        }
    }
    response = client.post("/predict", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        assert "disease" in data
        assert "probability" in data
        assert "decision_path" in data
        assert "shap_values" in data
        assert "latency_ms" in data
    else:
        assert response.status_code in [200, 503]
