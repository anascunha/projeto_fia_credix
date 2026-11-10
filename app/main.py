import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response

from app.schemas import HealthResponse, PredictRequest, PredictResponse
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
    yield


app = FastAPI(
    title="Credix API",
    version="1.1.0",
    description="Score de risco de inadimplência para solicitações de crédito.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health(request: Request, response: Response):
    preditor = getattr(request.app.state, "preditor", None)
    if preditor is None:
        response.status_code = 503
        return HealthResponse(status="indisponivel", modelo_carregado=False)
    return HealthResponse(status="ok", modelo_carregado=True, faixas=preditor.faixas)


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request):
    preditor = getattr(request.app.state, "preditor", None)
    if preditor is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    return preditor.prever(payload)
