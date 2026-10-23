"""Treina, compara e avalia os modelos; salva o melhor em ``models/``.

Uso:
    python -m churn.train            # gera dados, treina e salva
    python -m churn.train --n 2000   # versão rápida
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from . import __version__
from .data import FEATURES, TARGET, generate_customers
from .model import best_threshold, candidate_models, expected_profit

ROOT = Path(__file__).resolve().parents[2]


def train(n: int = 10_000, seed: int = 42, out_dir: Path = ROOT) -> dict:
    df = generate_customers(n=n, seed=seed)
    X, y = df[FEATURES], df[TARGET].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    comparison: dict[str, dict] = {}
    oof: dict[str, np.ndarray] = {}
    for name, pipe in candidate_models(seed).items():
        proba = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        oof[name] = proba
        comparison[name] = {
            "roc_auc_cv": round(roc_auc_score(y_train, proba), 4),
            "pr_auc_cv": round(average_precision_score(y_train, proba), 4),
        }
        print(f"{name:22s} ROC-AUC (CV) = {comparison[name]['roc_auc_cv']:.4f}")

    best_name = max(comparison, key=lambda k: comparison[k]["roc_auc_cv"])
    # O limiar é escolhido nas previsões fora da amostra do treino, nunca no teste.
    threshold, _ = best_threshold(y_train, oof[best_name], X_train["valor_mensal"].to_numpy())

    model = clone(candidate_models(seed)[best_name]).fit(X_train, y_train)
    proba_test = model.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= threshold).astype(int)
    monthly_test = X_test["valor_mensal"].to_numpy()

    importance = permutation_importance(
        model, X_test, y_test, scoring="roc_auc", n_repeats=5, random_state=seed
    )
    top_features = (
        pd.Series(importance.importances_mean, index=FEATURES)
        .sort_values(ascending=False)
        .round(4)
    )

    metrics = {
        "versao_modelo": __version__,
        "treinado_em": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_clientes": n,
        "taxa_churn": round(float(y.mean()), 4),
        "comparacao_cv": comparison,
        "modelo_escolhido": best_name,
        "limiar_decisao": threshold,
        "teste": {
            "roc_auc": round(roc_auc_score(y_test, proba_test), 4),
            "pr_auc": round(average_precision_score(y_test, proba_test), 4),
            "brier": round(brier_score_loss(y_test, proba_test), 4),
            "precisao": round(precision_score(y_test, pred_test), 4),
            "recall": round(recall_score(y_test, pred_test), 4),
            "clientes_abordados_pct": round(float(pred_test.mean()), 4),
            "lucro_campanha_modelo": round(
                expected_profit(y_test, proba_test, threshold, monthly_test), 2
            ),
            "lucro_campanha_todos": round(
                expected_profit(y_test, np.ones_like(proba_test), 0.5, monthly_test), 2
            ),
        },
        "importancia_permutacao": top_features.to_dict(),
    }

    (out_dir / "models").mkdir(exist_ok=True)
    (out_dir / "reports").mkdir(exist_ok=True)
    joblib.dump(
        {"pipeline": model, "threshold": threshold, "features": FEATURES, "version": __version__},
        out_dir / "models" / "model.joblib",
    )
    (out_dir / "reports" / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_figures(y_test, proba_test, monthly_test, threshold, top_features,
                 out_dir / "reports" / "figures")
    print(f"\nModelo escolhido: {best_name} | limiar = {threshold:.2f}")
    print(json.dumps(metrics["teste"], indent=2, ensure_ascii=False))
    return metrics


def save_figures(
    y_test: np.ndarray,
    proba: np.ndarray,
    monthly: np.ndarray,
    threshold: float,
    importance: pd.Series,
    folder: Path,
) -> None:
    """Gera os gráficos usados no README."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)

    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    profits = [expected_profit(y_test, proba, t, monthly) / 1000 for t in grid]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, profits, color="#2563eb", linewidth=2)
    ax.axvline(threshold, color="#dc2626", linestyle="--",
               label=f"limiar escolhido ({threshold:.2f})")
    ax.set_xlabel("Limiar de probabilidade para oferecer retenção")
    ax.set_ylabel("Lucro esperado (R$ mil)")
    ax.set_title("Lucro da campanha de retenção por limiar (conjunto de teste)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(folder / "lucro_por_limiar.png", dpi=150)
    plt.close(fig)

    imp = importance.sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(imp.index, imp.values, color="#2563eb")
    ax.set_xlabel("Queda média no ROC-AUC ao embaralhar a variável")
    ax.set_title("Importância das variáveis (permutação)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(folder / "importancia_variaveis.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10_000, help="número de clientes")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(n=args.n, seed=args.seed)


if __name__ == "__main__":
    main()
