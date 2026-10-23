"""API REST que serve o modelo de churn.

Rodar localmente:
    uvicorn churn.api:app --reload
Documentação interativa: http://localhost:8000/docs
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .model import risk_band, to_frame

MODEL_PATH = Path(
    os.getenv("MODEL_PATH", Path(__file__).resolve().parents[2] / "models" / "model.joblib")
)


class Customer(BaseModel):
    idade: int = Field(ge=18, le=100, examples=[34])
    tempo_cliente_meses: int = Field(ge=0, le=600, examples=[5])
    valor_mensal: float = Field(gt=0, examples=[154.8])
    chamados_suporte_90d: int = Field(ge=0, examples=[3])
    atrasos_pagamento_12m: int = Field(ge=0, examples=[2])
    tipo_contrato: Literal["mensal", "anual", "bianual"]
    forma_pagamento: Literal["boleto", "cartao_credito", "pix", "debito_automatico"]
    internet: Literal["fibra", "dsl", "nenhuma"]
    regiao: Literal["norte", "nordeste", "centro_oeste", "sudeste", "sul"]
    tem_streaming: bool
    tem_fidelidade_movel: bool


class Prediction(BaseModel):
    probabilidade_churn: float
    risco: Literal["baixo", "medio", "alto"]
    recomendar_oferta_retencao: bool


class BatchRequest(BaseModel):
    clientes: list[Customer] = Field(min_length=1, max_length=1000)


@lru_cache(maxsize=1)
def load_model() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {MODEL_PATH}. Rode `python -m churn.train` antes."
        )
    return joblib.load(MODEL_PATH)


app = FastAPI(
    title="API de Previsão de Churn",
    description="Estima a probabilidade de um cliente cancelar o serviço.",
    version=__version__,
)


def _predict(customers: list[Customer]) -> list[Prediction]:
    try:
        bundle = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    records = [{**c.model_dump(), "tem_streaming": int(c.tem_streaming),
                "tem_fidelidade_movel": int(c.tem_fidelidade_movel)} for c in customers]
    proba = bundle["pipeline"].predict_proba(to_frame(records)[bundle["features"]])[:, 1]
    return [
        Prediction(
            probabilidade_churn=round(float(p), 4),
            risco=risk_band(float(p)),
            recomendar_oferta_retencao=bool(p >= bundle["threshold"]),
        )
        for p in proba
    ]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "modelo_carregado": MODEL_PATH.exists(), "versao": __version__}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer) -> Prediction:
    return _predict([customer])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(request: BatchRequest) -> list[Prediction]:
    return _predict(request.clientes)
