"""Download dos dados: testado com um cliente falso da API do Kaggle (sem rede nem token)."""
import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "DataPipeline"))  # o script importa `utils`
_spec = importlib.util.spec_from_file_location("baixar_dados", RAIZ / "DataPipeline" / "00_baixar_dados.py")
baixar_dados = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baixar_dados)


class ApiFalsa:
    """Imita KaggleApi.competition_download_file: grava <arquivo>.zip no destino."""

    def __init__(self):
        self.chamadas = []

    def competition_download_file(self, competition, file_name, path=None, force=False, quiet=True):
        self.chamadas.append((competition, file_name))
        with zipfile.ZipFile(Path(path) / f"{file_name}.zip", "w") as zf:
            zf.writestr(file_name, "SK_ID_CURR,TARGET\n100002,1\n")


def test_baixa_e_descompacta_os_arquivos_do_projeto(tmp_path):
    api = ApiFalsa()
    caminhos = baixar_dados.baixar(api, destino=tmp_path)

    assert api.chamadas == [
        ("home-credit-default-risk", "application_train.csv"),
        ("home-credit-default-risk", "bureau.csv"),
    ]
    assert [c.name for c in caminhos] == ["application_train.csv", "bureau.csv"]
    assert all(c.read_text().startswith("SK_ID_CURR") for c in caminhos)
    assert not list(tmp_path.glob("*.zip"))


def test_nao_baixa_de_novo_arquivo_existente(tmp_path):
    (tmp_path / "application_train.csv").write_text("ja existe")
    api = ApiFalsa()
    baixar_dados.baixar(api, destino=tmp_path)
    assert api.chamadas == [("home-credit-default-risk", "bureau.csv")]
    assert (tmp_path / "application_train.csv").read_text() == "ja existe"

    baixar_dados.baixar(api, destino=tmp_path, forcar=True)
    assert len(api.chamadas) == 3


def test_erro_se_o_download_nao_gera_o_csv(tmp_path):
    class ApiQueNaoGrava(ApiFalsa):
        def competition_download_file(self, *a, **k):
            pass

    with pytest.raises(FileNotFoundError):
        baixar_dados.baixar(ApiQueNaoGrava(), destino=tmp_path, arquivos=["bureau.csv"])
