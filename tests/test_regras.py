import numpy as np
import pytest

from credix.regras import aplicar_regras, calcular_faixas, classificar_score, resumo_faixas


@pytest.mark.parametrize(
    "renda, comprometimento, regra",
    [
        (0, 0.0, "renda_nao_positiva"),
        (-10, 0.0, "renda_nao_positiva"),
        (100_000, 1.0, "parcela_maior_ou_igual_renda"),
        (100_000, 1.5, "parcela_maior_ou_igual_renda"),
    ],
)
def test_regras_que_decidem_sem_modelo(renda, comprometimento, regra):
    resultado = aplicar_regras(renda, comprometimento)
    assert resultado.nivel_risco == "alto"
    assert resultado.regra == regra


def test_sem_regra_o_modelo_decide():
    assert aplicar_regras(100_000, 0.99) is None


@pytest.mark.parametrize(
    "score, nivel",
    [(0.0, "baixo"), (0.299, "baixo"), (0.3, "medio"), (0.599, "medio"), (0.6, "alto"), (1.0, "alto")],
)
def test_classificar_score_nos_limites(score, nivel):
    assert classificar_score(score, {"medio": 0.3, "alto": 0.6}) == nivel


def test_faixas_pelos_quantis_do_teste():
    scores = np.linspace(0, 1, 101)
    faixas = calcular_faixas(scores)
    assert faixas == pytest.approx({"medio": 0.5, "alto": 0.8})
    niveis = [classificar_score(s, faixas) for s in scores]
    assert niveis.count("alto") == 21 and niveis.count("baixo") == 50


def test_resumo_faixas_mostra_taxa_por_nivel():
    scores = np.array([0.1, 0.2, 0.5, 0.9])
    y = np.array([0, 0, 1, 1])
    resumo = resumo_faixas(scores, y, {"medio": 0.4, "alto": 0.8})
    assert resumo.loc["baixo", "taxa_inadimplencia"] == 0
    assert resumo.loc["alto", "taxa_inadimplencia"] == 1
    assert resumo["participacao"].sum() == pytest.approx(1)
