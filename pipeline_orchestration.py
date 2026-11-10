"""Executa o pipeline completo localmente, sem Airflow (mesmas etapas da DAG)."""
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

ETAPAS = [
    BASE_DIR / "DataPipeline" / "01_application_features.py",
    BASE_DIR / "DataPipeline" / "02_bureau_features.py",
    BASE_DIR / "DataPipeline" / "merge_abt.py",
    BASE_DIR / "Model" / "train.py",
]

if __name__ == "__main__":
    print("=== Iniciando Pipeline Credix ===")
    for etapa in ETAPAS:
        print(f"--> {etapa.relative_to(BASE_DIR)}")
        subprocess.run([sys.executable, str(etapa)], check=True)
    print("=== Pipeline finalizado ===")
