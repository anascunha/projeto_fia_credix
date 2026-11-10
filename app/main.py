import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request, Response

from app.registro import criar_registro
from app.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    Predicao,
    ResultadoObservado,
)
from app.service import carregar_preditor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("credix.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.preditor = carregar_preditor()
    except Exception:
        log.exception("Falha ao carregar o modelo; /predict ficará indisponível")
        app.state.preditor = None
    try:
        app.state.registro = criar_registro()
    except Exception:
        log.exception("Falha ao abrir o registro de previsões; as previsões não serão gravadas")
        app.state.registro = None
    yield


app = FastAPI(
    title="Credix API",
    version="1.2.0",
    description=(
        "Score de risco de inadimplência para solicitações de crédito. "
        "Cada previsão é registrada e pode ser consultada em /predicoes."
    ),
    lifespan=lifespan,
)


def _preditor(request: Request):
    preditor = getattr(request.app.state, "preditor", None)
    if preditor is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    return preditor


def _registro(request: Request):
    registro = getattr(request.app.state, "registro", None)
    if registro is None:
        raise HTTPException(status_code=503, detail="Registro de previsões indisponível")
    return registro


@app.get("/health", response_model=HealthResponse)
def health(request: Request, response: Response):
    preditor = getattr(request.app.state, "preditor", None)
    registro_ativo = getattr(request.app.state, "registro", None) is not None
    if preditor is None:
        response.status_code = 503
        return HealthResponse(status="indisponivel", modelo_carregado=False, registro_ativo=registro_ativo)
    return HealthResponse(
        status="ok",
        modelo_carregado=True,
        versao_modelo=preditor.versao,
        registro_ativo=registro_ativo,
        faixas=preditor.faixas,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request):
    preditor = _preditor(request)
    resultado, features = preditor.prever_com_features(payload)

    registro = getattr(request.app.state, "registro", None)
    id_solicitacao = None
    if registro is not None:
        try:
            id_solicitacao = registro.salvar(
                entrada=payload.model_dump(exclude={"id_cliente"}),
                features=features,
                resultado=resultado,
                versao_modelo=preditor.versao,
                id_cliente=payload.id_cliente,
            )["id_solicitacao"]
        except Exception:
            # A previsão não deve falhar por causa do registro.
            log.exception("Falha ao registrar a previsão")
    return {"id_solicitacao": id_solicitacao, **resultado}


@app.get("/predicoes", response_model=list[Predicao])
def listar_predicoes(
    request: Request,
    limite: int = Query(50, ge=1, le=1000),
    nivel_risco: Literal["baixo", "medio", "alto"] | None = None,
    id_cliente: str | None = None,
    sem_resultado: bool = Query(False, description="Só previsões ainda sem resultado observado"),
):
    """Previsões registradas, da mais recente para a mais antiga."""
    return _registro(request).listar(
        limite=limite, nivel_risco=nivel_risco, id_cliente=id_cliente, sem_resultado=sem_resultado
    )


@app.get("/predicoes/{id_solicitacao}", response_model=Predicao)
def obter_predicao(id_solicitacao: str, request: Request):
    predicao = _registro(request).obter(id_solicitacao)
    if predicao is None:
        raise HTTPException(status_code=404, detail="Previsão não encontrada")
    return predicao


@app.post("/predicoes/{id_solicitacao}/resultado", response_model=Predicao)
def registrar_resultado(id_solicitacao: str, resultado: ResultadoObservado, request: Request):
    """Registra o comportamento real do cliente após o período de maturação."""
    predicao = _registro(request).registrar_resultado(id_solicitacao, resultado.inadimplente)
    if predicao is None:
        raise HTTPException(status_code=404, detail="Previsão não encontrada")
    return predicao
