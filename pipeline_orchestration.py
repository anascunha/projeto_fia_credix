import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

print("=== Iniciando Pipeline Credix ===")

subprocess.run(
    ["python", str(BASE_DIR / "DataPipeline" / "01_application_features.py")],
    check=True,
)

subprocess.run(
    ["python", str(BASE_DIR / "DataPipeline" / "02_bureau_features.py")],
    check=True,
)

subprocess.run(
    ["python", str(BASE_DIR / "DataPipeline" / "merge_abt.py")],
    check=True,
)

subprocess.run(
    ["python", str(BASE_DIR / "Model" / "train.py")],
    check=True,
)

print("=== Pipeline finalizado ===")