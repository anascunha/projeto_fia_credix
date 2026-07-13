import os
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier

from config import *

# df = pd.read_csv("Dados/abt.csv")

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(BASE_DIR / "Dados" / "abt.csv")

#separa x e y 
X = df.drop(columns=[TARGET, ID])
y = df[TARGET]

#categorias
categoricas = X.select_dtypes(include=["object"]).columns.tolist()

#pre processamento 
preprocessador = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categoricas
        )
    ],
    remainder="passthrough"
)

#treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

pipeline = Pipeline(
    steps=[
        (
            "preprocessamento",
            preprocessador
        ),
        (
            "modelo",
            XGBClassifier(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=12,   # exemplo
                random_state=RANDOM_STATE,
                eval_metric="logloss"
            )
        )
    ]
)

pipeline.fit(
    X_train,
    y_train
)

print("\nCOLUNAS UTILIZADAS NO TREINAMENTO:\n")
print(X_train.columns.tolist())

#probabilidade
pred = pipeline.predict(X_test)
prob = pipeline.predict_proba(X_test)[:,1]

#metricas
metricas = {
    "accuracy": accuracy_score(y_test,pred),
    "precision": precision_score(y_test,pred),
    "recall": recall_score(y_test,pred),
    "f1": f1_score(y_test,pred),
    "roc_auc": roc_auc_score(y_test,prob)
}

print("\n===== XGBoost =====")
for chave, valor in metricas.items():
    print(f"{chave}: {valor:.4f}")


models_dir = BASE_DIR / "models"
metrics_dir = BASE_DIR / "metrics"

models_dir.mkdir(exist_ok=True)
metrics_dir.mkdir(exist_ok=True)

joblib.dump(
    pipeline,
    models_dir / "best_model.pkl"
)

with open(
    metrics_dir / "train_metrics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(metricas, f, indent=4)

# joblib.dump(
#     pipeline,
#     "models/best_model.pkl"
# )

# with open(
#     "metrics/train_metrics.json",
#     "w",
#     encoding="utf-8"
# ) as f:
#     json.dump(
#         metricas,
#         f,
#         indent=4
#     )

# import pandas as pd

# feature_names = (
#     pipeline.named_steps["preprocessamento"]
#     .get_feature_names_out()
# )

# importancias = pd.DataFrame({
#     "feature": feature_names,
#     "importance": pipeline.named_steps["modelo"].feature_importances_
# })

# importancias = importancias.sort_values(
#     "importance",
#     ascending=False
# )

# print(importancias.head(20))
