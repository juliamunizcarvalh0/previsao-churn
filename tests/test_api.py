import pytest
from fastapi.testclient import TestClient

CLIENTE_RISCO = {
    "idade": 29,
    "tempo_cliente_meses": 2,
    "valor_mensal": 189.9,
    "chamados_suporte_90d": 5,
    "atrasos_pagamento_12m": 3,
    "tipo_contrato": "mensal",
    "forma_pagamento": "boleto",
    "internet": "fibra",
    "regiao": "sudeste",
    "tem_streaming": True,
    "tem_fidelidade_movel": False,
}
CLIENTE_FIEL = {
    **CLIENTE_RISCO,
    "idade": 55,
    "tempo_cliente_meses": 60,
    "valor_mensal": 79.9,
    "chamados_suporte_90d": 0,
    "atrasos_pagamento_12m": 0,
    "tipo_contrato": "bianual",
    "forma_pagamento": "debito_automatico",
    "internet": "dsl",
    "tem_streaming": False,
    "tem_fidelidade_movel": True,
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
    resp = client.post("/predict", json={**CLIENTE_RISCO, "tipo_contrato": "semanal"})
    assert resp.status_code == 422


def test_missing_model_returns_503(monkeypatch, tmp_path):
    from churn import api

    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "nao_existe.joblib")
    api.load_model.cache_clear()
    resp = TestClient(api.app).post("/predict", json=CLIENTE_RISCO)
    assert resp.status_code == 503
    api.load_model.cache_clear()
