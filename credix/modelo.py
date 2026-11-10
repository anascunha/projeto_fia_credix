"""Construção, treino e avaliação do modelo XGBoost do Credix."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from credix.features import COLUNAS_MODELO
from credix.regras import calcular_faixas, resumo_faixas

TARGET = "TARGET"
ID = "SK_ID_CURR"

# scale_pos_weight ~ razão entre negativos e positivos no treino.
# Na base Home Credit: 91,93% adimplentes / 8,07% inadimplentes ~ 11,4
# (ver DataPipeline/exp_analysis.ipynb). O valor 12 é esse número arredondado.
PARAMS_XGB = dict(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=12,
    eval_metric="logloss",
)


def construir_pipeline(categoricas: list[str], random_state: int = 42, **params) -> Pipeline:
    preprocessador = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categoricas)],
        remainder="passthrough",
    )
    modelo = XGBClassifier(**{**PARAMS_XGB, **params}, random_state=random_state)
    return Pipeline(steps=[("preprocessamento", preprocessador), ("modelo", modelo)])


def razao_classes(y: pd.Series) -> float:
    """Razão negativos/positivos -- referência para o ``scale_pos_weight``."""
    positivos = int((y == 1).sum())
    return float((y == 0).sum() / positivos) if positivos else float("nan")


def importancia_features(pipeline: Pipeline) -> pd.DataFrame:
    nomes = pipeline.named_steps["preprocessamento"].get_feature_names_out()
    nomes = [n.split("__", 1)[-1] for n in nomes]
    importancias = pipeline.named_steps["modelo"].feature_importances_
    return (
        pd.DataFrame({"feature": nomes, "importancia": importancias})
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )


def treinar_e_avaliar(
    abt: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    **params,
) -> dict:
    """Treina o pipeline na ABT e devolve modelo, métricas, faixas e importâncias."""
    X = abt[COLUNAS_MODELO]
    y = abt[TARGET]
    categoricas = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = construir_pipeline(categoricas, random_state=random_state, **params)
    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    score = pipeline.predict_proba(X_test)[:, 1]
    score_treino = pipeline.predict_proba(X_train)[:, 1]

    metricas = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, score),
        "roc_auc_treino": roc_auc_score(y_train, score_treino),
        "taxa_inadimplencia_teste": float(np.mean(y_test)),
        "score_medio_teste": float(np.mean(score)),
        "razao_classes_treino": razao_classes(y_train),
        "scale_pos_weight": pipeline.named_steps["modelo"].get_params()["scale_pos_weight"],
    }

    faixas = calcular_faixas(score)
    resumo = resumo_faixas(score, y_test.to_numpy(), faixas)
    metricas["faixas"] = resumo.reset_index().to_dict(orient="records")

    return {
        "pipeline": pipeline,
        "metricas": metricas,
        "faixas": faixas,
        "importancia": importancia_features(pipeline),
    }
