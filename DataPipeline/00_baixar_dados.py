"""Etapa 0: baixa os dados da competição Home Credit Default Risk (Kaggle).

Baixa só os arquivos usados pelo projeto para Dados/raw_data/:
    application_train.csv  (~160 MB)
    bureau.csv             (~170 MB)

Pré-requisitos (uma vez só):
    1. Ter conta no Kaggle e aceitar as regras da competição em
       https://www.kaggle.com/competitions/home-credit-default-risk/rules
    2. Autenticar a API do Kaggle, de uma destas formas:
       - `kaggle auth login` (login pelo navegador), ou
       - variável de ambiente KAGGLE_API_TOKEN com o token gerado em
         https://www.kaggle.com/settings/api, ou
       - o token salvo em ~/.kaggle/access_token

Uso:
    python DataPipeline/00_baixar_dados.py            # pula arquivos já baixados
    python DataPipeline/00_baixar_dados.py --forcar   # baixa de novo
"""
import argparse
import zipfile
from pathlib import Path

from utils import DADOS, RAW_DATA

COMPETICAO = "home-credit-default-risk"
ARQUIVOS = ["application_train.csv", "bureau.csv"]
DESTINO = DADOS / RAW_DATA


def _extrair_zip(destino: Path, arquivo: str) -> None:
    """A API entrega cada CSV compactado (arquivo.csv.zip); extrai e remove o zip."""
    zip_path = destino / f"{arquivo}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(destino)
        zip_path.unlink()


def baixar(api, destino: Path = DESTINO, arquivos=ARQUIVOS, forcar: bool = False) -> list[Path]:
    """Baixa os CSVs da competição com um cliente da API do Kaggle já autenticado."""
    destino.mkdir(parents=True, exist_ok=True)
    caminhos = []
    for arquivo in arquivos:
        csv = destino / arquivo
        if csv.exists() and not forcar:
            print(f"{arquivo}: já existe, pulando (use --forcar para baixar de novo)")
        else:
            print(f"{arquivo}: baixando...")
            api.competition_download_file(COMPETICAO, arquivo, path=str(destino), force=forcar, quiet=False)
            _extrair_zip(destino, arquivo)
            if not csv.exists():
                raise FileNotFoundError(f"Download de {arquivo} não gerou {csv}")
        caminhos.append(csv)
    return caminhos


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--forcar", action="store_true", help="baixa de novo mesmo se o arquivo existir")
    args = parser.parse_args()

    # Importado aqui: o pacote kaggle tenta autenticar já no import.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    try:
        caminhos = baixar(api, forcar=args.forcar)
    except Exception as erro:
        if "403" in str(erro) or "Forbidden" in str(erro):
            raise SystemExit(
                "Acesso negado (403). Aceite as regras da competição em "
                f"https://www.kaggle.com/competitions/{COMPETICAO}/rules e rode de novo."
            ) from erro
        raise
    for csv in caminhos:
        print(f"ok: {csv} ({csv.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
