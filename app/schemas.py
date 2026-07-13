from pydantic import BaseModel

class PredictRequest(BaseModel):

    idade: float

    escolaridade: str

    estado_civil: str

    renda_anual: float

    tempo_empresa: float

    valor_credito: float

    valor_bem: float

    valor_parcela: float

    qtd_creditos: float

    qtd_creditos_ativos: float

    valor_total_credito: float

    valor_total_divida: float



class PredictResponse(BaseModel):

    probabilidade_inadimplencia: float

    nivel_risco: str