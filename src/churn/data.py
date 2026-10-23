"""Geração de um dataset sintético e realista de clientes de uma operadora.

Os dados são sintéticos para que o projeto rode em qualquer máquina sem
download. As relações entre as variáveis imitam padrões conhecidos do setor:
contratos mensais, muitos chamados no suporte e pagamento por boleto aumentam
a chance de cancelamento; clientes antigos e com contrato longo cancelam menos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CONTRACT_TYPES = ["mensal", "anual", "bianual"]
PAYMENT_METHODS = ["boleto", "cartao_credito", "pix", "debito_automatico"]
INTERNET_TYPES = ["fibra", "dsl", "nenhuma"]
REGIONS = ["norte", "nordeste", "centro_oeste", "sudeste", "sul"]

NUMERIC_FEATURES = [
    "idade",
    "tempo_cliente_meses",
    "valor_mensal",
    "chamados_suporte_90d",
    "atrasos_pagamento_12m",
]
CATEGORICAL_FEATURES = ["tipo_contrato", "forma_pagamento", "internet", "regiao"]
BINARY_FEATURES = ["tem_streaming", "tem_fidelidade_movel"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
TARGET = "churn"


def generate_customers(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Gera ``n`` clientes sintéticos com a coluna alvo ``churn`` (0/1)."""
    rng = np.random.default_rng(seed)

    contract = rng.choice(CONTRACT_TYPES, size=n, p=[0.55, 0.30, 0.15])
    payment = rng.choice(PAYMENT_METHODS, size=n, p=[0.25, 0.30, 0.25, 0.20])
    internet = rng.choice(INTERNET_TYPES, size=n, p=[0.50, 0.35, 0.15])
    region = rng.choice(REGIONS, size=n, p=[0.08, 0.27, 0.08, 0.42, 0.15])

    age = rng.normal(40, 13, n).clip(18, 85).round().astype(int)

    # Contratos longos tendem a pertencer a clientes mais antigos.
    tenure_scale = np.select(
        [contract == "mensal", contract == "anual"], [14, 30], default=45
    )
    tenure = rng.exponential(tenure_scale).clip(1, 96).round().astype(int)

    streaming = rng.binomial(1, 0.45, n)
    mobile_bundle = rng.binomial(1, 0.30, n)

    base_price = np.select(
        [internet == "fibra", internet == "dsl"], [119.9, 79.9], default=49.9
    )
    monthly = (
        base_price + streaming * 34.9 + rng.normal(0, 12, n)
    ).clip(29.9, 299.9).round(2)

    support_calls = rng.poisson(np.where(internet == "fibra", 1.2, 0.8), n)
    late_payments = rng.poisson(np.where(payment == "boleto", 1.0, 0.35), n)

    logit = (
        -2.2
        + 1.30 * (contract == "mensal")
        - 0.90 * (contract == "bianual")
        - 0.035 * tenure
        + 0.012 * (monthly - 100)
        + 0.38 * support_calls
        + 0.30 * late_payments
        + 0.45 * (payment == "boleto")
        - 0.35 * (payment == "debito_automatico")
        + 0.35 * (internet == "fibra")
        - 0.55 * mobile_bundle
        - 0.008 * (age - 40)
        + rng.normal(0, 0.35, n)  # ruído não observável
    )
    churn = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    df = pd.DataFrame(
        {
            "cliente_id": [f"CLI{i:06d}" for i in range(n)],
            "idade": age,
            "tempo_cliente_meses": tenure,
            "valor_mensal": monthly,
            "chamados_suporte_90d": support_calls,
            "atrasos_pagamento_12m": late_payments,
            "tipo_contrato": contract,
            "forma_pagamento": payment,
            "internet": internet,
            "regiao": region,
            "tem_streaming": streaming,
            "tem_fidelidade_movel": mobile_bundle,
            TARGET: churn,
        }
    )
    return df


if __name__ == "__main__":
    from pathlib import Path

    out = Path("data")
    out.mkdir(exist_ok=True)
    data = generate_customers()
    data.to_csv(out / "clientes.csv", index=False)
    print(f"{len(data)} clientes salvos em data/clientes.csv "
          f"(taxa de churn: {data[TARGET].mean():.1%})")
