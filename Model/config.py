import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Permite importar o pacote compartilhado `credix` ao rodar os scripts diretamente.
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

RANDOM_STATE = 42

TEST_SIZE = 0.20

TARGET = "TARGET"

ID = "SK_ID_CURR"

ABT = BASE_DIR / "Dados" / "abt.csv"

MODEL_FOLDER = BASE_DIR / "models"

METRICS_FOLDER = BASE_DIR / "metrics"
