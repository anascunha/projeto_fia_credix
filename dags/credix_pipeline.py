"""DAG do Credix: uma tarefa por etapa do pipeline de dados e treino.

application_features ─┐
                      ├─> construir_abt ─> treinar_modelo
bureau_features ──────┘

As features cadastrais e de bureau são independentes e rodam em paralelo.
O treino gera modelo, faixas de risco e métricas (ver Model/train.py).
"""
from datetime import datetime

try:  # Airflow 3
    from airflow.providers.standard.operators.bash import BashOperator
    from airflow.sdk import DAG
except ImportError:  # Airflow 2
    from airflow import DAG
    from airflow.operators.bash import BashOperator

PROJETO = "/opt/airflow"

with DAG(
    dag_id="credix_pipeline",
    description="Features -> ABT -> treino do XGBoost",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["credix"],
) as dag:

    application_features = BashOperator(
        task_id="application_features",
        bash_command=f"python {PROJETO}/DataPipeline/01_application_features.py",
    )

    bureau_features = BashOperator(
        task_id="bureau_features",
        bash_command=f"python {PROJETO}/DataPipeline/02_bureau_features.py",
    )

    construir_abt = BashOperator(
        task_id="construir_abt",
        bash_command=f"python {PROJETO}/DataPipeline/merge_abt.py",
    )

    treinar_modelo = BashOperator(
        task_id="treinar_modelo",
        bash_command=f"python {PROJETO}/Model/train.py",
    )

    [application_features, bureau_features] >> construir_abt >> treinar_modelo
