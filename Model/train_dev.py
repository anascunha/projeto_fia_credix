import os

os.makedirs("models", exist_ok=True)
os.makedirs("metrics", exist_ok=True)

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier

from config import *

df = pd.read_csv("Dados/abt.csv")

X = df.drop(columns=[TARGET, ID])
y = df[TARGET]

categoricas = X.select_dtypes(include=["object"]).columns.tolist()
numericas = X.select_dtypes(exclude=["object"]).columns.tolist()
print(categoricas)

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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

pipeline_lr = Pipeline(
    steps=[
        ("preprocessamento", preprocessador),
        (
            "modelo",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )
    ]
)

pipeline_lr.fit(
    X_train,
    y_train
)

pred_lr = pipeline_lr.predict(X_test)
prob_lr = pipeline_lr.predict_proba(X_test)[:,1]

print("===== Logistic Regression =====")

print(
    "Accuracy:",
    accuracy_score(y_test,pred_lr)
)

print(
    "Precision:",
    precision_score(y_test,pred_lr)
)

print(
    "Recall:",
    recall_score(y_test,pred_lr)
)

print(
    "F1:",
    f1_score(y_test,pred_lr)
)

print(
    "ROC AUC:",
    roc_auc_score(y_test,prob_lr)
)

joblib.dump(
    pipeline_lr,
    "models/logistic.pkl"
)

pipeline_rf = Pipeline(
    steps=[
        ("preprocessamento", preprocessador),
        (
            "modelo",
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )
    ]
)

pipeline_rf.fit(
    X_train,
    y_train
)

pred_rf = pipeline_rf.predict(X_test)
prob_rf = pipeline_rf.predict_proba(X_test)[:,1]

print("===== Random Forest =====")

print(
    "Accuracy:",
    accuracy_score(y_test,pred_rf)
)

print(
    "Precision:",
    precision_score(y_test,pred_rf)
)

print(
    "Recall:",
    recall_score(y_test,pred_rf)
)

print(
    "F1:",
    f1_score(y_test,pred_rf)
)

print(
    "ROC AUC:",
    roc_auc_score(y_test,prob_rf)
)

joblib.dump(
    pipeline_rf,
    "models/random_forest.pkl"
)

pipeline_xgb = Pipeline(
    steps=[
        ("preprocessamento", preprocessador),
        (
            "modelo",
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                eval_metric="logloss"
            )
        )
    ]
)

pipeline_xgb.fit(
    X_train,
    y_train
)

pred_xgb = pipeline_xgb.predict(X_test)
prob_xgb = pipeline_xgb.predict_proba(X_test)[:,1]

print("===== xgb =====")

print(
    "Accuracy:",
    accuracy_score(y_test,pred_xgb)
)

print(
    "Precision:",
    precision_score(y_test,pred_xgb)
)

print(
    "Recall:",
    recall_score(y_test,pred_xgb)
)

print(
    "F1:",
    f1_score(y_test,pred_xgb)
)

print(
    "ROC AUC:",
    roc_auc_score(y_test,prob_xgb)
)

joblib.dump(
    pipeline_xgb,
    "models/xgboost.pkl"
)