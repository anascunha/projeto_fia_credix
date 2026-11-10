"""Etapa 3: construção da ABT (application + bureau)."""
import pandas as pd

from utils import ABT, APPLICATION_FEATURES, BUREAU_FEATURES

from credix.features import montar_abt


def main():
    abt = montar_abt(pd.read_csv(APPLICATION_FEATURES), pd.read_csv(BUREAU_FEATURES))
    abt.to_csv(ABT, index=False)
    print(f"ABT {abt.shape} -> {ABT}")


if __name__ == "__main__":
    main()
