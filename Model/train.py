"""Treino do modelo final (XGBoost) a partir da ABT.

Gera:
- models/best_model.pkl          pipeline (pré-processamento + XGBoost)
- models/faixas_risco.json       cortes de score das faixas baixo/médio/alto
- metrics/train_metrics.json     métricas no conjunto de teste
- metrics/importancia_features.csv
"""
import json
import logging

import joblib
import pandas as pd

from config import ABT, METRICS_FOLDER, MODEL_FOLDER, RANDOM_STATE, TEST_SIZE

from credix.modelo import treinar_e_avaliar

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("credix.train")


def main():
    abt = pd.read_csv(ABT)
    resultado = treinar_e_avaliar(abt, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    metricas = resultado["metricas"]

    log.info("===== XGBoost (teste) =====")
    for chave in ["accuracy", "precision", "recall", "f1", "roc_auc", "roc_auc_treino"]:
        log.info("%s: %.4f", chave, metricas[chave])
    log.info(
        "razão negativos/positivos no treino: %.2f (scale_pos_weight=%s)",
        metricas["razao_classes_treino"],
        metricas["scale_pos_weight"],
    )
    log.info("faixas de score: %s", resultado["faixas"])
    for faixa in metricas["faixas"]:
        log.info(
            "  %-5s %5.1f%% dos clientes | inadimplência observada %.1f%%",
            faixa["nivel_risco"],
            100 * faixa["participacao"],
            100 * faixa["taxa_inadimplencia"],
        )

    MODEL_FOLDER.mkdir(exist_ok=True)
    METRICS_FOLDER.mkdir(exist_ok=True)

    joblib.dump(resultado["pipeline"], MODEL_FOLDER / "best_model.pkl")
    with open(MODEL_FOLDER / "faixas_risco.json", "w", encoding="utf-8") as f:
        json.dump(resultado["faixas"], f, indent=4)
    with open(METRICS_FOLDER / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=4, ensure_ascii=False)
    resultado["importancia"].to_csv(METRICS_FOLDER / "importancia_features.csv", index=False)

    log.info("modelo salvo em %s", MODEL_FOLDER / "best_model.pkl")


if __name__ == "__main__":
    main()
