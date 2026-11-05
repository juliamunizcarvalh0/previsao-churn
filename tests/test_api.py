import pytest
from fastapi.testclient import TestClient

CLIENTE_RISCO = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.5,
    "TotalCharges": 191.0,
}
CLIENTE_FIEL = {
    **CLIENTE_RISCO,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 60,
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 55.0,
    "TotalCharges": 3300.0,
}


@pytest.fixture()
def client(trained_dir, monkeypatch):
    from churn import api

    out, _ = trained_dir
    monkeypatch.setattr(api, "MODEL_PATH", out / "models" / "model.joblib")
    api.load_model.cache_clear()
    yield TestClient(api.app)
    api.load_model.cache_clear()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["modelo_carregado"] is True


def test_predict_ranks_risky_customer_higher(client):
    risky = client.post("/predict", json=CLIENTE_RISCO).json()
    loyal = client.post("/predict", json=CLIENTE_FIEL).json()
    assert risky["probabilidade_churn"] > loyal["probabilidade_churn"]
    assert risky["recomendar_oferta_retencao"] is True
    assert loyal["risco"] == "baixo"


def test_batch(client):
    resp = client.post("/predict/batch", json={"clientes": [CLIENTE_RISCO, CLIENTE_FIEL]})
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_invalid_input_returns_422(client):
    resp = client.post("/predict", json={**CLIENTE_RISCO, "Contract": "Weekly"})
    assert resp.status_code == 422


def test_missing_model_returns_503(monkeypatch, tmp_path):
    from churn import api

    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "nao_existe.joblib")
    api.load_model.cache_clear()
    resp = TestClient(api.app).post("/predict", json=CLIENTE_RISCO)
    assert resp.status_code == 503
    api.load_model.cache_clear()
