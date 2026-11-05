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


YesNo = Literal["Yes", "No"]
InternetDependent = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    """Cliente no mesmo formato do dataset IBM Telco Customer Churn."""

    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1, examples=[0])
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=600, examples=[2], description="meses como cliente")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetDependent
    OnlineBackup: InternetDependent
    DeviceProtection: InternetDependent
    TechSupport: InternetDependent
    StreamingTV: InternetDependent
    StreamingMovies: InternetDependent
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(gt=0, examples=[95.5])
    TotalCharges: float = Field(ge=0, examples=[191.0])


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
    description="Estima a probabilidade de um cliente cancelar o serviço. "
    "Modelo treinado no dataset público IBM Telco Customer Churn.",
    version=__version__,
)


def _predict(customers: list[Customer]) -> list[Prediction]:
    try:
        bundle = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    records = [c.model_dump() for c in customers]
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
