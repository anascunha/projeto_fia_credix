"""Carregamento do modelo e lógica de predição da API."""
import hashlib
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
    def __init__(self, modelo, faixas: dict[str, float], versao: str | None = None):
        self.modelo = modelo
        self.faixas = faixas
        self.versao = versao

    def calcular_features(self, request: PredictRequest):
        return features_solicitacao(**request.model_dump(exclude={"id_cliente"}))

    def prever(self, request: PredictRequest) -> dict:
        """Score, nível de risco e regra aplicada para uma solicitação."""
        return self.prever_com_features(request)[0]

    def prever_com_features(self, request: PredictRequest) -> tuple[dict, dict]:
        """Como :meth:`prever`, devolvendo também as features enviadas ao modelo."""
        entrada = self.calcular_features(request)
        features = entrada.iloc[0].to_dict()

        regra = aplicar_regras(
            renda_mensal=request.renda_mensal,
            comprometimento_renda=float(entrada["COMPROMETIMENTO_RENDA"].iloc[0]),
        )
        if regra is not None:
            resultado = {"score_risco": None, "nivel_risco": regra.nivel_risco, "regra_aplicada": regra.regra}
            return resultado, features

        score = float(self.modelo.predict_proba(entrada)[0][1])
        resultado = {
            "score_risco": round(score, 4),
            "nivel_risco": classificar_score(score, self.faixas),
            "regra_aplicada": None,
        }
        return resultado, features


def versao_do_modelo(model_path: Path) -> str:
    """Identifica o modelo pelo hash do arquivo: muda a cada retreino."""
    return hashlib.sha256(Path(model_path).read_bytes()).hexdigest()[:12]


def carregar_preditor(model_path: Path = MODEL_PATH, faixas_path: Path = FAIXAS_PATH) -> Preditor:
    modelo = joblib.load(model_path)
    if faixas_path.exists():
        faixas = json.loads(faixas_path.read_text(encoding="utf-8"))
    else:
        log.warning("%s não encontrado; usando faixas padrão %s", faixas_path, FAIXAS_PADRAO)
        faixas = FAIXAS_PADRAO
    versao = versao_do_modelo(model_path)
    log.info("Modelo %s carregado de %s (faixas: %s)", versao, model_path, faixas)
    return Preditor(modelo, faixas, versao)
