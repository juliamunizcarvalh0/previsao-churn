from churn.data import FEATURES, TARGET, generate_customers


def test_shape_and_columns():
    df = generate_customers(n=500, seed=1)
    assert len(df) == 500
    assert set(FEATURES + [TARGET, "cliente_id"]) == set(df.columns)
    assert df["cliente_id"].is_unique


def test_reproducible_with_same_seed():
    a = generate_customers(n=200, seed=3)
    b = generate_customers(n=200, seed=3)
    assert a.equals(b)


def test_values_are_plausible():
    df = generate_customers(n=5000, seed=2)
    assert df["idade"].between(18, 85).all()
    assert (df["valor_mensal"] > 0).all()
    assert 0.10 < df[TARGET].mean() < 0.40


def test_monthly_contracts_churn_more():
    df = generate_customers(n=8000, seed=4)
    rate = df.groupby("tipo_contrato")[TARGET].mean()
    assert rate["mensal"] > rate["anual"] > rate["bianual"]
