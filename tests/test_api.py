import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.service import carregar_preditor
from credix.features import features_solicitacao

PAYLOAD = {
    "idade": 42,
    "escolaridade": "Higher education",
    "estado_civil": "Married",
    "renda_anual": 250000,
    "tempo_empresa": 12,
    "valor_credito": 180000,
    "valor_bem": 220000,
    "valor_parcela_anual": 42000,
    "qtd_creditos": 4,
    "qtd_creditos_ativos": 2,
    "valor_total_credito": 600000,
    "valor_total_divida": 50000,
}


@pytest.fixture
def client(modelo_treinado, monkeypatch):
    pasta = modelo_treinado["pasta"]
    monkeypatch.setattr(
        main, "carregar_preditor",
        lambda: carregar_preditor(pasta / "best_model.pkl", pasta / "faixas_risco.json"),
    )
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def client_sem_modelo(tmp_path, monkeypatch):
    monkeypatch.setattr(
        main, "carregar_preditor",
        lambda: carregar_preditor(tmp_path / "nao_existe.pkl", tmp_path / "faixas.json"),
    )
    with TestClient(main.app) as c:
        yield c


def test_health_ok(client, modelo_treinado):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "modelo_carregado": True, "faixas": modelo_treinado["faixas"]}


def test_health_sem_modelo_retorna_503(client_sem_modelo):
    r = client_sem_modelo.get("/health")
    assert r.status_code == 503
    assert r.json()["modelo_carregado"] is False
    assert client_sem_modelo.post("/predict", json=PAYLOAD).status_code == 503


def test_predict_usa_o_modelo_com_as_features_do_treino(client, modelo_treinado):
    r = client.post("/predict", json=PAYLOAD)
    assert r.status_code == 200
    corpo = r.json()

    esperado = modelo_treinado["pipeline"].predict_proba(features_solicitacao(**PAYLOAD))[0][1]
    assert corpo["score_risco"] == pytest.approx(round(float(esperado), 4))
    assert corpo["nivel_risco"] in {"baixo", "medio", "alto"}
    assert corpo["regra_aplicada"] is None


@pytest.mark.parametrize(
    "alteracao, regra",
    [
        ({"valor_parcela_anual": 250000}, "parcela_maior_ou_igual_renda"),
        ({"valor_parcela_anual": 300000}, "parcela_maior_ou_igual_renda"),
        ({"renda_anual": 0}, "renda_nao_positiva"),
    ],
)
def test_regras_de_negocio_na_api(client, alteracao, regra):
    r = client.post("/predict", json={**PAYLOAD, **alteracao})
    assert r.status_code == 200
    assert r.json() == {"score_risco": None, "nivel_risco": "alto", "regra_aplicada": regra}


@pytest.mark.parametrize(
    "alteracao",
    [
        {"renda_anual": -1},
        {"idade": 10},
        {"escolaridade": "Doutorado"},
        {"qtd_creditos": 1, "qtd_creditos_ativos": 2},
        {"valor_parcela": 3500},  # nome antigo do campo, sem a unidade
    ],
)
def test_validacao_da_entrada(client, alteracao):
    payload = {**PAYLOAD, **alteracao}
    if "valor_parcela" in alteracao:
        payload.pop("valor_parcela_anual")
    assert client.post("/predict", json=payload).status_code == 422
