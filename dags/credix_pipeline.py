from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="credix_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["credix"],
) as dag:

    executar_pipeline = BashOperator(
        task_id="executar_pipeline",
        bash_command="cd /opt/airflow && python pipeline_orchestration.py"
    )