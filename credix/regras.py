"""Regras de negócio e faixas de risco do Credix.

O modelo usa ``scale_pos_weight`` para compensar o desbalanceamento das
classes, o que desloca o ``predict_proba`` para cima. Por isso a saída do
modelo é tratada como **score de risco** (0 a 1, ordena clientes do menor
para o maior risco), e não como probabilidade calibrada de inadimplência.

As faixas baixo/médio/alto são definidas a partir da distribuição do score no
conjunto de teste (ver :func:`calcular_faixas`) e salvas em
``models/faixas_risco.json`` pelo ``Model/train.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Proporção de clientes de teste em cada faixa, do menor para o maior score.
# 50% baixo, 30% médio, 20% alto.
QUANTIS_FAIXAS = {"medio": 0.50, "alto": 0.80}

# Cortes usados quando ``faixas_risco.json`` não existe (valores que já
# estavam no código original do projeto).
FAIXAS_PADRAO = {"medio": 0.20, "alto": 0.50}


@dataclass(frozen=True)
class ResultadoRegra:
    nivel_risco: str
    regra: str


def aplicar_regras(renda_mensal: float, comprometimento_renda: float) -> ResultadoRegra | None:
    """Regras que decidem o risco sem consultar o modelo.

    ``comprometimento_renda`` é ``parcela / renda`` na mesma periodicidade
    (a feature ``COMPROMETIMENTO_RENDA``). Retorna ``None`` quando nenhuma
    regra se aplica e o modelo deve ser consultado.
    """
    if renda_mensal <= 0:
        return ResultadoRegra("alto", "renda_nao_positiva")
    if comprometimento_renda >= 1:
        return ResultadoRegra("alto", "parcela_maior_ou_igual_renda")
    return None


def classificar_score(score: float, faixas: dict[str, float]) -> str:
    """Converte o score do modelo em nível de risco usando os cortes informados."""
    if score >= faixas["alto"]:
        return "alto"
    if score >= faixas["medio"]:
        return "medio"
    return "baixo"


def calcular_faixas(scores: np.ndarray, quantis: dict[str, float] = QUANTIS_FAIXAS) -> dict[str, float]:
    """Define os cortes das faixas pelos quantis do score no conjunto de teste."""
    return {nome: float(np.quantile(scores, q)) for nome, q in quantis.items()}


def resumo_faixas(scores: np.ndarray, y_true: np.ndarray, faixas: dict[str, float]) -> pd.DataFrame:
    """Taxa de inadimplência observada em cada faixa (evidência de que as faixas ordenam o risco)."""
    niveis = [classificar_score(s, faixas) for s in scores]
    df = pd.DataFrame({"nivel_risco": niveis, "inadimplente": np.asarray(y_true)})
    resumo = df.groupby("nivel_risco")["inadimplente"].agg(clientes="size", taxa_inadimplencia="mean")
    resumo["participacao"] = resumo["clientes"] / len(df)
    return resumo.reindex(["baixo", "medio", "alto"]).fillna(0)
