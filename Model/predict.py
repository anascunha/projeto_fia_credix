"""Exemplo: score de um cliente da ABT usando o modelo treinado."""
import json

import joblib
import pandas as pd

from config import ABT, ID, MODEL_FOLDER, TARGET

from credix.features import COLUNAS_MODELO
from credix.regras import FAIXAS_PADRAO, classificar_score


def main():
    modelo = joblib.load(MODEL_FOLDER / "best_model.pkl")
    caminho_faixas = MODEL_FOLDER / "faixas_risco.json"
    faixas = json.loads(caminho_faixas.read_text()) if caminho_faixas.exists() else FAIXAS_PADRAO

    cliente = pd.read_csv(ABT).iloc[[0]]
    score = float(modelo.predict_proba(cliente[COLUNAS_MODELO])[0][1])

    print({
        ID: int(cliente[ID].iloc[0]),
        TARGET: int(cliente[TARGET].iloc[0]),
        "score_risco": round(score, 4),
        "nivel_risco": classificar_score(score, faixas),
    })


if __name__ == "__main__":
    main()
