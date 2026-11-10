import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.registro import RegistroPredicoes
from app.service import carregar_preditor
from credix.features import features_solicitacao

PAYLOAD = {
    "idade": 42,
    "escolaridade": "Higher education",
    "estado_civil": "Married",
    "renda_mensal": 250000,
    "tempo_empresa": 12,
    "valor_credito": 180000,
    "valor_bem": 220000,
    "valor_parcela_mensal": 42000,
    "qtd_creditos": 4,
    "qtd_creditos_ativos": 2,
    "valor_total_credito": 600000,
    "valor_total_divida": 50000,
}


@pytest.fixture(autouse=True)
def registro_temporario(tmp_path, monkeypatch):
    """Nenhum teste grava no banco real (registros/predicoes.db)."""
    monkeypatch.setattr(main, "criar_registro", lambda: RegistroPredicoes(tmp_path / "predicoes.db"))


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
    corpo = r.json()
    assert corpo["status"] == "ok" and corpo["modelo_carregado"] and corpo["registro_ativo"]
    assert corpo["faixas"] == modelo_treinado["faixas"]
    assert len(corpo["versao_modelo"]) == 12


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
        ({"valor_parcela_mensal": 250000}, "parcela_maior_ou_igual_renda"),
        ({"valor_parcela_mensal": 300000}, "parcela_maior_ou_igual_renda"),
        ({"renda_mensal": 0}, "renda_nao_positiva"),
    ],
)
def test_regras_de_negocio_na_api(client, alteracao, regra):
    r = client.post("/predict", json={**PAYLOAD, **alteracao})
    assert r.status_code == 200
    corpo = r.json()
    assert corpo.pop("id_solicitacao")
    assert corpo == {"score_risco": None, "nivel_risco": "alto", "regra_aplicada": regra}


@pytest.mark.parametrize(
    "alteracao",
    [
        {"renda_mensal": -1},
        {"idade": 10},
        {"escolaridade": "Doutorado"},
        {"qtd_creditos": 1, "qtd_creditos_ativos": 2},
        {"valor_parcela": 3500},  # nome antigo do campo, sem a unidade
    ],
)
def test_validacao_da_entrada(client, alteracao):
    payload = {**PAYLOAD, **alteracao}
    if "valor_parcela" in alteracao:
        payload.pop("valor_parcela_mensal")
    assert client.post("/predict", json=payload).status_code == 422


def test_previsao_fica_registrada_com_entrada_e_features(client):
    r = client.post("/predict", json={**PAYLOAD, "id_cliente": "cliente-42"})
    corpo = r.json()

    reg = client.get(f"/predicoes/{corpo['id_solicitacao']}").json()
    assert reg["id_cliente"] == "cliente-42"
    assert reg["score_risco"] == corpo["score_risco"]
    assert reg["nivel_risco"] == corpo["nivel_risco"]
    assert reg["entrada"]["renda_mensal"] == PAYLOAD["renda_mensal"]
    assert "id_cliente" not in reg["entrada"]
    assert reg["features"]["COMPROMETIMENTO_RENDA"] == pytest.approx(42000 / 250000)
    assert len(reg["features"]) == 21
    assert reg["versao_modelo"] == client.get("/health").json()["versao_modelo"]
    assert reg["resultado_observado"] is None


def test_previsao_decidida_por_regra_tambem_e_registrada(client):
    corpo = client.post("/predict", json={**PAYLOAD, "valor_parcela_mensal": 300000}).json()
    reg = client.get(f"/predicoes/{corpo['id_solicitacao']}").json()
    assert reg["regra_aplicada"] == "parcela_maior_ou_igual_renda"
    assert reg["score_risco"] is None


def test_listar_filtrar_e_registrar_resultado(client):
    ids = [client.post("/predict", json={**PAYLOAD, "id_cliente": f"c{i}"}).json()["id_solicitacao"] for i in range(3)]
    client.post("/predict", json={**PAYLOAD, "renda_mensal": 0, "id_cliente": "c-regra"})

    todas = client.get("/predicoes").json()
    assert len(todas) == 4
    assert todas[0]["id_cliente"] == "c-regra"  # mais recente primeiro
    assert [p["id_cliente"] for p in client.get("/predicoes", params={"id_cliente": "c1"}).json()] == ["c1"]
    assert len(client.get("/predicoes", params={"limite": 2}).json()) == 2
    assert all(p["nivel_risco"] == "alto" for p in client.get("/predicoes", params={"nivel_risco": "alto"}).json())

    r = client.post(f"/predicoes/{ids[0]}/resultado", json={"inadimplente": True})
    assert r.status_code == 200
    assert r.json()["resultado_observado"] is True and r.json()["data_resultado"]
    pendentes = client.get("/predicoes", params={"sem_resultado": True}).json()
    assert ids[0] not in {p["id_solicitacao"] for p in pendentes} and len(pendentes) == 3


def test_previsao_inexistente_retorna_404(client):
    assert client.get("/predicoes/nao-existe").status_code == 404
    assert client.post("/predicoes/nao-existe/resultado", json={"inadimplente": False}).status_code == 404


def test_predict_continua_funcionando_sem_registro(client, monkeypatch):
    main.app.state.registro = None
    r = client.post("/predict", json=PAYLOAD)
    assert r.status_code == 200 and r.json()["id_solicitacao"] is None
    assert client.get("/predicoes").status_code == 503
    assert client.get("/health").json()["registro_ativo"] is False
