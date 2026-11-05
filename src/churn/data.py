"""Carga e limpeza do dataset público IBM Telco Customer Churn.

O arquivo ``data/Telco-Customer-Churn.csv`` vem do repositório oficial da IBM
(https://github.com/IBM/telco-customer-churn-on-icp4d, licença Apache 2.0) e é
o mesmo dataset publicado no Kaggle como "Telco Customer Churn". São 7.043
clientes de uma operadora de telecom, com 26,5% de cancelamento.

Os nomes das colunas foram mantidos como no original para que o código rode
igual no Kaggle, no GitHub e na API.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_FEATURES = ["SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
TARGET = "Churn"
ID = "customerID"


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Corrige os tipos do CSV original.

    ``TotalCharges`` vem como texto e fica em branco para os 11 clientes com
    ``tenure == 0``: eles ainda não pagaram nenhuma fatura, então o total é 0.
    """
    df = raw.copy()
    total = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    df["TotalCharges"] = total.fillna(0.0).astype(float)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    df[TARGET] = (df[TARGET].astype(str) == "Yes").astype(int)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str)
    return df[[ID, *FEATURES, TARGET]]


def load_telco(path: Path | str | None = None) -> pd.DataFrame:
    """Lê o CSV (local ou, se não existir, do repositório da IBM) já limpo."""
    path = Path(path) if path else DATA_PATH
    source = path if path.exists() else SOURCE_URL
    return clean(pd.read_csv(source))


if __name__ == "__main__":
    data = load_telco()
    print(f"{len(data)} clientes | taxa de churn: {data[TARGET].mean():.1%}")
