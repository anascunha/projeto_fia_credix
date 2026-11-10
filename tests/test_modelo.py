from credix.features import COLUNAS_MODELO


def test_treino_gera_metricas_faixas_e_importancias(modelo_treinado):
    m = modelo_treinado["metricas"]
    for chave in ["accuracy", "precision", "recall", "f1", "roc_auc", "roc_auc_treino", "faixas"]:
        assert chave in m
    # sinal artificial nos dados sintéticos: o modelo precisa ordenar melhor que o acaso
    assert m["roc_auc"] > 0.6

    faixas = modelo_treinado["faixas"]
    assert 0 <= faixas["medio"] < faixas["alto"] <= 1

    taxas = {f["nivel_risco"]: f["taxa_inadimplencia"] for f in m["faixas"]}
    assert taxas["baixo"] < taxas["alto"]

    imp = modelo_treinado["importancia"]
    assert abs(imp["importancia"].sum() - 1) < 1e-6
    assert set(COLUNAS_MODELO) - {"NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS"} <= set(imp["feature"])
