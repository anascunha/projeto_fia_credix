"""Carregamento do modelo e lógica de predição da API."""
import json
import logging
import os
from pathlib import Path

import joblib

from app.schemas import PredictRequest
from credix.features import features_solicitacao
from credix.regras import FAIXAS_PADRAO, aplicar_regras, classificar_score

log = logging.getLogger("credix.api")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("CREDIX_MODEL_PATH", BASE_DIR / "models" / "best_model.pkl"))
FAIXAS_PATH = Path(os.getenv("CREDIX_FAIXAS_PATH", MODEL_PATH.parent / "faixas_risco.json"))


class Preditor:
    def __init__(self, modelo, faixas: dict[str, float]):
        self.modelo = modelo
        self.faixas = faixas

    def prever(self, request: PredictRequest) -> dict:
        entrada = features_solicitacao(**request.model_dump())

        regra = aplicar_regras(
            renda_anual=request.renda_anual,
            comprometimento_renda=float(entrada["COMPROMETIMENTO_RENDA"].iloc[0]),
        )
        if regra is not None:
            return {"score_risco": None, "nivel_risco": regra.nivel_risco, "regra_aplicada": regra.regra}

        score = float(self.modelo.predict_proba(entrada)[0][1])
        return {
            "score_risco": round(score, 4),
            "nivel_risco": classificar_score(score, self.faixas),
            "regra_aplicada": None,
        }


def carregar_preditor(model_path: Path = MODEL_PATH, faixas_path: Path = FAIXAS_PATH) -> Preditor:
    modelo = joblib.load(model_path)
    if faixas_path.exists():
        faixas = json.loads(faixas_path.read_text(encoding="utf-8"))
    else:
        log.warning("%s não encontrado; usando faixas padrão %s", faixas_path, FAIXAS_PADRAO)
        faixas = FAIXAS_PADRAO
    log.info("Modelo carregado de %s (faixas: %s)", model_path, faixas)
    return Preditor(modelo, faixas)
