import numpy as np

from churn.model import best_threshold, expected_profit, risk_band


def test_training_beats_random_and_saves_artifacts(trained_dir):
    out, metrics = trained_dir
    assert metrics["teste"]["roc_auc"] > 0.80  # literatura do dataset: ~0,84
    assert (out / "models" / "model.joblib").exists()
    assert (out / "reports" / "metrics.json").exists()
    assert (out / "reports" / "figures" / "lucro_por_limiar.png").exists()


def test_model_campaign_beats_contacting_everyone(trained_dir):
    _, metrics = trained_dir
    t = metrics["teste"]
    assert t["lucro_campanha_modelo"] > t["lucro_campanha_todos"]


def test_expected_profit_counts_costs():
    y = np.array([0, 0, 1])
    proba = np.array([0.9, 0.9, 0.1])
    monthly = np.array([100.0, 100.0, 100.0])
    # Aborda dois clientes que não iriam sair: só custo.
    assert expected_profit(y, proba, 0.5, monthly) < 0


def test_best_threshold_is_in_range():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, 1000)
    proba = np.clip(y * 0.5 + rng.uniform(0, 0.5, 1000), 0, 1)
    threshold, profit = best_threshold(y, proba, np.full(1000, 120.0))
    assert 0.05 <= threshold <= 0.95
    assert profit > 0


def test_risk_band():
    assert risk_band(0.1) == "baixo"
    assert risk_band(0.45) == "medio"
    assert risk_band(0.8) == "alto"
