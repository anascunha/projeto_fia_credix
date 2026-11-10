"""Engenharia de atributos do Credix: fonte única para treino e API.

As mesmas funções são usadas por:

- ``DataPipeline/01_application_features.py`` e ``02_bureau_features.py`` (treino);
- ``app/service.py`` (predição), via :func:`features_solicitacao`.

Assim, uma feature calculada no treino é calculada exatamente da mesma forma
na API. O teste ``tests/test_features.py`` garante essa equivalência.

Unidades
--------
O modelo é treinado com as colunas originais da Home Credit:

- ``AMT_INCOME_TOTAL``: renda total do cliente, tratada como **anual**;
- ``AMT_ANNUITY``: valor da parcela do empréstimo, tratado na **mesma
  periodicidade da renda** (anual).

As razões ``PARCELA_RENDA``/``COMPROMETIMENTO_RENDA`` e a regra de negócio só
fazem sentido se parcela e renda estiverem na mesma unidade. Por isso a API
recebe ``renda_anual`` e ``valor_parcela_anual`` (ver ``app/schemas.py``) e
os repassa sem conversão para ``AMT_INCOME_TOTAL`` e ``AMT_ANNUITY``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Valor sentinela da Home Credit para "sem vínculo empregatício" (~1000 anos).
DAYS_EMPLOYED_SENTINELA = 365243

# Crédito acima de N vezes a renda é marcado como "crédito alto".
LIMITE_CREDITO_ALTO = 5

COLUNAS_APPLICATION = [
    "SK_ID_CURR",
    "TARGET",
    "AMT_INCOME_TOTAL",  # renda anual do cliente
    "AMT_CREDIT",  # valor do crédito solicitado
    "AMT_ANNUITY",  # valor da parcela (mesma periodicidade da renda)
    "AMT_GOODS_PRICE",  # valor do bem financiado
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
]

COLUNAS_BUREAU = [
    "BUREAU_QTD_CREDITOS",
    "BUREAU_QTD_ATIVOS",
    "BUREAU_TOTAL_CREDITO",
    "BUREAU_TOTAL_DIVIDA",
    "BUREAU_DEBT_RATIO",
]

# Ordem exata das colunas de entrada do modelo (ABT sem TARGET e SK_ID_CURR).
COLUNAS_MODELO = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "IDADE",
    "TEMPO_EMPRESA",
    "CREDITO_RENDA",
    "PARCELA_RENDA",
    "BEM_RENDA",
    "RENDA_LIVRE",
    "COMPROMETIMENTO_RENDA",
    "PARCELA_MAIOR_RENDA",
    "CREDITO_ALTO",
    "PRAZO_ESTIMADO",
    *COLUNAS_BUREAU,
]

_RAZOES = ["CREDITO_RENDA", "PARCELA_RENDA", "BEM_RENDA"]
_COLUNAS_ARREDONDADAS = ["IDADE", "TEMPO_EMPRESA", *_RAZOES, "PRAZO_ESTIMADO"]


def adicionar_features_financeiras(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula as features derivadas de renda, crédito, parcela e bem.

    Espera as colunas ``AMT_INCOME_TOTAL``, ``AMT_CREDIT``, ``AMT_ANNUITY``,
    ``AMT_GOODS_PRICE``, ``IDADE`` e ``TEMPO_EMPRESA`` (estas duas em anos).
    Retorna uma cópia com as colunas novas.
    """
    df = df.copy()
    renda = df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    parcela = df["AMT_ANNUITY"]

    df["CREDITO_RENDA"] = df["AMT_CREDIT"] / renda
    df["PARCELA_RENDA"] = parcela / renda
    df["BEM_RENDA"] = df["AMT_GOODS_PRICE"] / renda

    df["RENDA_LIVRE"] = df["AMT_INCOME_TOTAL"] - parcela
    df["COMPROMETIMENTO_RENDA"] = (parcela / renda).fillna(0)
    df["PARCELA_MAIOR_RENDA"] = (parcela > df["AMT_INCOME_TOTAL"]).astype(int)
    df["CREDITO_ALTO"] = (df["CREDITO_RENDA"] > LIMITE_CREDITO_ALTO).astype(int)

    df[_RAZOES] = df[_RAZOES].fillna(0).clip(lower=0)

    df["PRAZO_ESTIMADO"] = (df["AMT_CREDIT"] / parcela.replace(0, np.nan)).fillna(0)

    df[_COLUNAS_ARREDONDADAS] = df[_COLUNAS_ARREDONDADAS].round(2)
    return df


def calcular_debt_ratio(total_divida: pd.Series, total_credito: pd.Series) -> pd.Series:
    """Razão dívida/crédito do bureau (crédito zero é tratado como 1)."""
    return total_divida / total_credito.replace(0, 1)


def preparar_application(application: pd.DataFrame) -> pd.DataFrame:
    """Transforma ``application_train.csv`` na base de features cadastrais."""
    df = application[COLUNAS_APPLICATION].copy()

    df["IDADE"] = df["DAYS_BIRTH"].abs() / 365

    dias_emprego = df["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_SENTINELA, np.nan)
    df["TEMPO_EMPRESA"] = (dias_emprego.abs() / 365).fillna(0)

    df = df.drop(columns=["DAYS_BIRTH", "DAYS_EMPLOYED"])
    df = adicionar_features_financeiras(df)

    assert df["SK_ID_CURR"].is_unique, "SK_ID_CURR duplicado na base de aplicações"
    return df


def agregar_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    """Agrega ``bureau.csv`` por cliente (``SK_ID_CURR``)."""
    grupos = bureau.assign(
        ATIVO=(bureau["CREDIT_ACTIVE"] == "Active").astype(int)
    ).groupby("SK_ID_CURR")

    features = pd.concat(
        [
            grupos.size().rename("BUREAU_QTD_CREDITOS"),
            grupos["ATIVO"].sum().rename("BUREAU_QTD_ATIVOS"),
            grupos["AMT_CREDIT_SUM"].sum().rename("BUREAU_TOTAL_CREDITO"),
            grupos["AMT_CREDIT_SUM_DEBT"].sum().rename("BUREAU_TOTAL_DIVIDA"),
        ],
        axis=1,
    ).reset_index()

    features["BUREAU_DEBT_RATIO"] = calcular_debt_ratio(
        features["BUREAU_TOTAL_DIVIDA"], features["BUREAU_TOTAL_CREDITO"]
    )
    return features


def montar_abt(application_features: pd.DataFrame, bureau_features: pd.DataFrame) -> pd.DataFrame:
    """Junta features cadastrais e de bureau; clientes sem bureau recebem 0."""
    abt = application_features.merge(bureau_features, on="SK_ID_CURR", how="left")
    return abt.fillna(0)


def features_solicitacao(
    *,
    idade: float,
    tempo_empresa: float,
    escolaridade: str,
    estado_civil: str,
    renda_anual: float,
    valor_credito: float,
    valor_bem: float,
    valor_parcela_anual: float,
    qtd_creditos: float,
    qtd_creditos_ativos: float,
    valor_total_credito: float,
    valor_total_divida: float,
) -> pd.DataFrame:
    """Monta a linha de entrada do modelo para uma solicitação da API.

    Usa as mesmas funções do pipeline de treino e devolve um DataFrame de uma
    linha com as colunas em :data:`COLUNAS_MODELO`.
    """
    df = pd.DataFrame(
        [
            {
                "AMT_INCOME_TOTAL": float(renda_anual),
                "AMT_CREDIT": float(valor_credito),
                "AMT_ANNUITY": float(valor_parcela_anual),
                "AMT_GOODS_PRICE": float(valor_bem),
                "NAME_EDUCATION_TYPE": escolaridade,
                "NAME_FAMILY_STATUS": estado_civil,
                "IDADE": float(idade),
                "TEMPO_EMPRESA": float(tempo_empresa),
                "BUREAU_QTD_CREDITOS": float(qtd_creditos),
                "BUREAU_QTD_ATIVOS": float(qtd_creditos_ativos),
                "BUREAU_TOTAL_CREDITO": float(valor_total_credito),
                "BUREAU_TOTAL_DIVIDA": float(valor_total_divida),
            }
        ]
    )
    df = adicionar_features_financeiras(df)
    df["BUREAU_DEBT_RATIO"] = calcular_debt_ratio(
        df["BUREAU_TOTAL_DIVIDA"], df["BUREAU_TOTAL_CREDITO"]
    )
    return df[COLUNAS_MODELO]
