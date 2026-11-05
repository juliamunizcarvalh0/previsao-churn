import pandas as pd

from churn.data import DATA_PATH, FEATURES, ID, TARGET, clean, load_telco


def test_dataset_shape_and_columns():
    df = load_telco()
    assert DATA_PATH.exists(), "o CSV da IBM deve estar versionado em data/"
    assert len(df) == 7043
    assert list(df.columns) == [ID, *FEATURES, TARGET]
    assert df[ID].is_unique


def test_types_and_no_missing_values():
    df = load_telco()
    assert df[FEATURES].notna().all().all()
    assert df["TotalCharges"].dtype == float
    assert set(df[TARGET].unique()) == {0, 1}
    assert 0.25 < df[TARGET].mean() < 0.28  # 26,5% no dataset original


def test_blank_total_charges_become_zero():
    raw = pd.read_csv(DATA_PATH)
    blank = raw["TotalCharges"].astype(str).str.strip() == ""
    assert blank.sum() == 11
    cleaned = clean(raw)
    assert (cleaned.loc[blank, "TotalCharges"] == 0).all()
    assert (cleaned.loc[blank, "tenure"] == 0).all()


def test_monthly_contracts_churn_more():
    df = load_telco()
    rate = df.groupby("Contract")[TARGET].mean()
    assert rate["Month-to-month"] > rate["One year"] > rate["Two year"]
