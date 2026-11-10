"""Dados SINTÉTICOS no formato de application_train.csv e bureau.csv.

Usados apenas nos testes: não substituem a base do Kaggle e não servem para
medir desempenho do modelo.
"""
import numpy as np
import pandas as pd

ESCOLARIDADES = [
    "Lower secondary",
    "Secondary / secondary special",
    "Incomplete higher",
    "Higher education",
    "Academic degree",
]
ESTADOS_CIVIS = ["Civil marriage", "Married", "Separated", "Single / not married", "Widow"]


def gerar_application(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    renda = rng.lognormal(mean=12, sigma=0.5, size=n).round(1)
    credito = (renda * rng.uniform(0.5, 8, size=n)).round(1)
    parcela = (credito / rng.uniform(8, 30, size=n)).round(1)
    bem = (credito * rng.uniform(0.8, 1.0, size=n)).round(1)
    dias_nasc = -rng.integers(21 * 365, 69 * 365, size=n)
    dias_emp = -rng.integers(0, 30 * 365, size=n)
    dias_emp[rng.random(n) < 0.15] = 365243  # sentinela "sem emprego"

    # risco latente: parcela alta vs renda, idade baixa
    logit = -3 + 4 * (parcela / renda) - 0.03 * (dias_nasc / -365 - 40)
    target = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(100000, 100000 + n),
            "TARGET": target,
            "NAME_CONTRACT_TYPE": "Cash loans",  # coluna extra, ignorada pelo pipeline
            "AMT_INCOME_TOTAL": renda,
            "AMT_CREDIT": credito,
            "AMT_ANNUITY": parcela,
            "AMT_GOODS_PRICE": bem,
            "DAYS_BIRTH": dias_nasc,
            "DAYS_EMPLOYED": dias_emp,
            "NAME_EDUCATION_TYPE": rng.choice(ESCOLARIDADES, size=n),
            "NAME_FAMILY_STATUS": rng.choice(ESTADOS_CIVIS, size=n),
        }
    )
    # casos de borda presentes na base real
    df.loc[df.index[:5], "AMT_ANNUITY"] = np.nan
    df.loc[df.index[5:10], "AMT_GOODS_PRICE"] = np.nan
    return df


def gerar_bureau(ids: np.ndarray, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    com_bureau = ids[rng.random(len(ids)) < 0.85]  # parte dos clientes sem histórico
    qtd = rng.integers(1, 8, size=len(com_bureau))
    sk = np.repeat(com_bureau, qtd)
    m = len(sk)
    credito = rng.lognormal(12, 1, size=m).round(1)
    credito[rng.random(m) < 0.03] = 0
    divida = (credito * rng.uniform(0, 1, size=m)).round(1)
    divida[rng.random(m) < 0.1] = np.nan
    return pd.DataFrame(
        {
            "SK_ID_CURR": sk,
            "SK_ID_BUREAU": np.arange(5000000, 5000000 + m),
            "CREDIT_ACTIVE": rng.choice(["Active", "Closed", "Sold"], size=m, p=[0.4, 0.55, 0.05]),
            "AMT_CREDIT_SUM": credito,
            "AMT_CREDIT_SUM_DEBT": divida,
        }
    )
