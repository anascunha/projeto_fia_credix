"""Etapa 2: agregação do histórico de crédito (bureau.csv) por cliente."""
import pandas as pd

from utils import BUREAU, BUREAU_FEATURES

from credix.features import agregar_bureau


def main():
    df = agregar_bureau(pd.read_csv(BUREAU))
    BUREAU_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(BUREAU_FEATURES, index=False)
    print(f"{len(df)} clientes -> {BUREAU_FEATURES}")


if __name__ == "__main__":
    main()
