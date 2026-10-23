"""Construção do pipeline, escolha do limiar de decisão e explicabilidade."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES

# Premissas de negócio (ajustáveis): oferecer um desconto de retenção custa
# R$ 50 por cliente abordado; perder um cliente custa, em média, 6 meses de
# receita. A oferta convence 35% dos clientes que iriam cancelar.
RETENTION_OFFER_COST = 50.0
LOST_CUSTOMER_MONTHS = 6
OFFER_SUCCESS_RATE = 0.35


def _preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ]
    )


def candidate_models(seed: int = 42) -> dict[str, Pipeline]:
    """Modelos comparados no treino: um baseline linear e um de árvores."""
    return {
        "regressao_logistica": Pipeline(
            [
                ("prep", _preprocessor(scale_numeric=True)),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("prep", _preprocessor(scale_numeric=False)),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=300,
                        max_leaf_nodes=15,
                        l2_regularization=1.0,
                        early_stopping=True,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def expected_profit(
    y_true: np.ndarray, proba: np.ndarray, threshold: float, monthly_value: np.ndarray
) -> float:
    """Lucro esperado de abordar os clientes com probabilidade >= limiar."""
    targeted = proba >= threshold
    saved_revenue = (
        y_true[targeted] * monthly_value[targeted] * LOST_CUSTOMER_MONTHS * OFFER_SUCCESS_RATE
    ).sum()
    cost = targeted.sum() * RETENTION_OFFER_COST
    return float(saved_revenue - cost)


def best_threshold(
    y_true: np.ndarray, proba: np.ndarray, monthly_value: np.ndarray
) -> tuple[float, float]:
    """Procura o limiar que maximiza o lucro esperado da campanha de retenção."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    profits = [expected_profit(y_true, proba, t, monthly_value) for t in grid]
    i = int(np.argmax(profits))
    return float(grid[i]), float(profits[i])


def risk_band(probability: float) -> str:
    if probability >= 0.6:
        return "alto"
    if probability >= 0.3:
        return "medio"
    return "baixo"


def to_frame(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)
