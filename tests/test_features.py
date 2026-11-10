"""A API precisa calcular as features exatamente como o pipeline de treino."""
import numpy as np
import pandas as pd
import pytest

from credix.features import (
    COLUNAS_MODELO,
    agregar_bureau,
    features_solicitacao,
    montar_abt,
    preparar_application,
)


def _solicitacao_a_partir_dos_dados_brutos(app_row, bureau_feat):
    """Monta o payload que um cliente da API enviaria para este registro."""
    dias_emp = app_row["DAYS_EMPLOYED"]
    b = bureau_feat.get(app_row["SK_ID_CURR"])
    return dict(
        idade=abs(app_row["DAYS_BIRTH"]) / 365,
        tempo_empresa=0.0 if dias_emp == 365243 else abs(dias_emp) / 365,
        escolaridade=app_row["NAME_EDUCATION_TYPE"],
        estado_civil=app_row["NAME_FAMILY_STATUS"],
        renda_anual=app_row["AMT_INCOME_TOTAL"],
        valor_credito=app_row["AMT_CREDIT"],
        valor_bem=app_row["AMT_GOODS_PRICE"],
        valor_parcela_anual=app_row["AMT_ANNUITY"],
        qtd_creditos=0 if b is None else b["BUREAU_QTD_CREDITOS"],
        qtd_creditos_ativos=0 if b is None else b["BUREAU_QTD_ATIVOS"],
        valor_total_credito=0 if b is None else b["BUREAU_TOTAL_CREDITO"],
        valor_total_divida=0 if b is None else b["BUREAU_TOTAL_DIVIDA"],
    )


def test_api_reproduz_as_features_da_abt(application_raw, bureau_raw, abt):
    bureau_feat = agregar_bureau(bureau_raw).set_index("SK_ID_CURR").to_dict(orient="index")
    completos = application_raw.dropna(subset=["AMT_ANNUITY", "AMT_GOODS_PRICE"])
    assert len(completos) > 100

    for _, linha in completos.iterrows():
        esperado = abt.loc[abt["SK_ID_CURR"] == linha["SK_ID_CURR"], COLUNAS_MODELO].reset_index(drop=True)
        obtido = features_solicitacao(**_solicitacao_a_partir_dos_dados_brutos(linha, bureau_feat))
        pd.testing.assert_frame_equal(obtido, esperado, check_dtype=False, rtol=1e-9)


def test_colunas_da_abt_sao_as_do_modelo(abt):
    assert [c for c in abt.columns if c not in ("SK_ID_CURR", "TARGET")] == COLUNAS_MODELO


def test_valores_calculados_a_mao():
    f = features_solicitacao(
        idade=30.123,
        tempo_empresa=2.5,
        escolaridade="Higher education",
        estado_civil="Married",
        renda_anual=100_000,
        valor_credito=600_000,
        valor_bem=500_000,
        valor_parcela_anual=30_000,
        qtd_creditos=3,
        qtd_creditos_ativos=1,
        valor_total_credito=200_000,
        valor_total_divida=50_000,
    ).iloc[0]
    assert f["IDADE"] == 30.12
    assert f["CREDITO_RENDA"] == 6.0
    assert f["PARCELA_RENDA"] == 0.3
    assert f["BEM_RENDA"] == 5.0
    assert f["RENDA_LIVRE"] == 70_000
    assert f["COMPROMETIMENTO_RENDA"] == pytest.approx(0.3)
    assert f["PARCELA_MAIOR_RENDA"] == 0
    assert f["CREDITO_ALTO"] == 1  # crédito > 5x a renda
    assert f["PRAZO_ESTIMADO"] == 20.0
    assert f["BUREAU_DEBT_RATIO"] == 0.25


def test_sentinela_de_emprego_vira_tempo_zero(application_raw):
    df = preparar_application(application_raw)
    sentinelas = application_raw["DAYS_EMPLOYED"] == 365243
    assert sentinelas.any()
    assert (df.loc[sentinelas.to_numpy(), "TEMPO_EMPRESA"] == 0).all()


def test_parcela_ausente_nao_gera_nulo_na_abt(abt, application_raw):
    ids = application_raw.loc[application_raw["AMT_ANNUITY"].isna(), "SK_ID_CURR"]
    linhas = abt[abt["SK_ID_CURR"].isin(ids)]
    assert not linhas.isna().any().any()
    assert (linhas["PRAZO_ESTIMADO"] == 0).all()


def test_bureau_credito_zero_e_cliente_sem_historico():
    bureau = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2],
            "CREDIT_ACTIVE": ["Active", "Closed", "Active"],
            "AMT_CREDIT_SUM": [0.0, 0.0, 1000.0],
            "AMT_CREDIT_SUM_DEBT": [300.0, np.nan, 250.0],
        }
    )
    feat = agregar_bureau(bureau).set_index("SK_ID_CURR")
    assert feat.loc[1, "BUREAU_QTD_CREDITOS"] == 2
    assert feat.loc[1, "BUREAU_QTD_ATIVOS"] == 1
    assert feat.loc[1, "BUREAU_DEBT_RATIO"] == 300.0  # crédito zero é tratado como 1
    assert feat.loc[2, "BUREAU_DEBT_RATIO"] == 0.25

    app = pd.DataFrame({"SK_ID_CURR": [1, 2, 3], "X": [1, 2, 3]})
    abt = montar_abt(app, feat.reset_index())
    assert (abt.loc[abt["SK_ID_CURR"] == 3, ["BUREAU_QTD_CREDITOS", "BUREAU_DEBT_RATIO"]] == 0).all().all()
