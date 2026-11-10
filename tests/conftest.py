import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from credix.features import agregar_bureau, montar_abt, preparar_application  # noqa: E402
from tests.dados_sinteticos import gerar_application, gerar_bureau  # noqa: E402


@pytest.fixture(scope="session")
def application_raw():
    return gerar_application()


@pytest.fixture(scope="session")
def bureau_raw(application_raw):
    return gerar_bureau(application_raw["SK_ID_CURR"].to_numpy())


@pytest.fixture(scope="session")
def abt(application_raw, bureau_raw):
    return montar_abt(preparar_application(application_raw), agregar_bureau(bureau_raw))


@pytest.fixture(scope="session")
def modelo_treinado(abt, tmp_path_factory):
    """Treina o pipeline real em dados sintéticos e salva em disco (como o train.py)."""
    import json

    import joblib

    from credix.modelo import treinar_e_avaliar

    resultado = treinar_e_avaliar(abt, n_estimators=30)
    pasta = tmp_path_factory.mktemp("models")
    joblib.dump(resultado["pipeline"], pasta / "best_model.pkl")
    (pasta / "faixas_risco.json").write_text(json.dumps(resultado["faixas"]))
    return {**resultado, "pasta": pasta}
