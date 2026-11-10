"""Registro das previsões em SQLite.

Cada chamada a ``POST /predict`` grava uma linha com a entrada, as features
calculadas, o resultado e a versão do modelo. O histórico serve para:

- consulta por um agente de IA (explicar decisões, simular contrapropostas);
- monitoramento: depois do período de maturação, o resultado real do cliente é
  registrado em ``resultado_observado`` e comparado com a previsão.

O banco fica em ``registros/predicoes.db`` (configurável por ``CREDIX_DB_PATH``).
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("CREDIX_DB_PATH", BASE_DIR / "registros" / "predicoes.db"))

_CRIAR_TABELA = """
CREATE TABLE IF NOT EXISTS predicoes (
    id_solicitacao       TEXT PRIMARY KEY,
    id_cliente           TEXT,
    data_previsao        TEXT NOT NULL,
    versao_modelo        TEXT,
    entrada              TEXT NOT NULL,   -- JSON da requisição
    features             TEXT NOT NULL,   -- JSON das 21 features enviadas ao modelo
    score_risco          REAL,
    nivel_risco          TEXT NOT NULL,
    regra_aplicada       TEXT,
    resultado_observado  INTEGER,         -- 1 = inadimplente, 0 = adimplente, NULL = em maturação
    data_resultado       TEXT
)
"""

_COLUNAS = [
    "id_solicitacao",
    "id_cliente",
    "data_previsao",
    "versao_modelo",
    "entrada",
    "features",
    "score_risco",
    "nivel_risco",
    "regra_aplicada",
    "resultado_observado",
    "data_resultado",
]


def _agora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _linha_para_dict(linha: tuple) -> dict:
    registro = dict(zip(_COLUNAS, linha))
    registro["entrada"] = json.loads(registro["entrada"])
    registro["features"] = json.loads(registro["features"])
    if registro["resultado_observado"] is not None:
        registro["resultado_observado"] = bool(registro["resultado_observado"])
    return registro


class RegistroPredicoes:
    def __init__(self, caminho: Path = DB_PATH):
        self.caminho = Path(caminho)
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._conectar()) as con, con:
            con.execute(_CRIAR_TABELA)

    def _conectar(self) -> sqlite3.Connection:
        # Uma conexão por operação: seguro com os workers/threads do FastAPI.
        return sqlite3.connect(self.caminho)

    def salvar(
        self,
        *,
        entrada: dict,
        features: dict,
        resultado: dict,
        versao_modelo: str | None,
        id_cliente: str | None = None,
    ) -> dict:
        registro = {
            "id_solicitacao": str(uuid.uuid4()),
            "id_cliente": id_cliente,
            "data_previsao": _agora(),
            "versao_modelo": versao_modelo,
        }
        with closing(self._conectar()) as con, con:
            con.execute(
                "INSERT INTO predicoes (id_solicitacao, id_cliente, data_previsao, versao_modelo,"
                " entrada, features, score_risco, nivel_risco, regra_aplicada)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    registro["id_solicitacao"],
                    id_cliente,
                    registro["data_previsao"],
                    versao_modelo,
                    json.dumps(entrada, ensure_ascii=False),
                    json.dumps(features, ensure_ascii=False),
                    resultado["score_risco"],
                    resultado["nivel_risco"],
                    resultado["regra_aplicada"],
                ),
            )
        return registro

    def obter(self, id_solicitacao: str) -> dict | None:
        with closing(self._conectar()) as con:
            linha = con.execute(
                f"SELECT {', '.join(_COLUNAS)} FROM predicoes WHERE id_solicitacao = ?",
                (id_solicitacao,),
            ).fetchone()
        return _linha_para_dict(linha) if linha else None

    def listar(
        self,
        *,
        limite: int = 50,
        nivel_risco: str | None = None,
        id_cliente: str | None = None,
        sem_resultado: bool = False,
    ) -> list[dict]:
        filtros, params = [], []
        if nivel_risco:
            filtros.append("nivel_risco = ?")
            params.append(nivel_risco)
        if id_cliente:
            filtros.append("id_cliente = ?")
            params.append(id_cliente)
        if sem_resultado:
            filtros.append("resultado_observado IS NULL")
        onde = f"WHERE {' AND '.join(filtros)}" if filtros else ""
        with closing(self._conectar()) as con:
            linhas = con.execute(
                f"SELECT {', '.join(_COLUNAS)} FROM predicoes {onde}"
                " ORDER BY data_previsao DESC, rowid DESC LIMIT ?",
                (*params, limite),
            ).fetchall()
        return [_linha_para_dict(linha) for linha in linhas]

    def registrar_resultado(self, id_solicitacao: str, inadimplente: bool) -> dict | None:
        with closing(self._conectar()) as con, con:
            cursor = con.execute(
                "UPDATE predicoes SET resultado_observado = ?, data_resultado = ? WHERE id_solicitacao = ?",
                (int(inadimplente), _agora(), id_solicitacao),
            )
        return self.obter(id_solicitacao) if cursor.rowcount else None


def criar_registro() -> RegistroPredicoes:
    return RegistroPredicoes(DB_PATH)
