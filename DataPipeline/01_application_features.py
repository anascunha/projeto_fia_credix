"""Etapa 1: features cadastrais e financeiras (application_train.csv).

A lógica de features fica em `credix/features.py`, compartilhada com a API.
"""
import pandas as pd

from utils import APPLICATION, APPLICATION_FEATURES

from credix.features import preparar_application


def main():
    df = preparar_application(pd.read_csv(APPLICATION))
    APPLICATION_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(APPLICATION_FEATURES, index=False)
    print(f"{len(df)} linhas -> {APPLICATION_FEATURES}")


if __name__ == "__main__":
    main()
