from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Categorias presentes na base Home Credit (application_train.csv).
Escolaridade = Literal[
    "Lower secondary",
    "Secondary / secondary special",
    "Incomplete higher",
    "Higher education",
    "Academic degree",
]

EstadoCivil = Literal[
    "Civil marriage",
    "Married",
    "Separated",
    "Single / not married",
    "Widow",
    "Unknown",
]


class PredictRequest(BaseModel):
    """Dados de uma solicitação de crédito.

    Valores monetários na mesma moeda da base de treino. Renda e parcela devem
    estar na MESMA periodicidade (anual), como no treino do modelo.
    """

    idade: float = Field(ge=18, le=100, description="Idade em anos")
    escolaridade: Escolaridade
    estado_civil: EstadoCivil
    renda_anual: float = Field(ge=0, description="Renda total anual (AMT_INCOME_TOTAL)")
    tempo_empresa: float = Field(ge=0, description="Tempo no emprego atual, em anos")
    valor_credito: float = Field(ge=0, description="Valor do crédito solicitado (AMT_CREDIT)")
    valor_bem: float = Field(ge=0, description="Valor do bem financiado (AMT_GOODS_PRICE)")
    valor_parcela_anual: float = Field(
        ge=0, description="Valor anual da parcela do empréstimo (AMT_ANNUITY)"
    )
    qtd_creditos: int = Field(ge=0, description="Créditos registrados no bureau")
    qtd_creditos_ativos: int = Field(ge=0, description="Créditos ativos no bureau")
    valor_total_credito: float = Field(ge=0, description="Soma dos créditos no bureau")
    valor_total_divida: float = Field(ge=0, description="Soma das dívidas no bureau")

    @model_validator(mode="after")
    def _ativos_nao_excedem_total(self):
        if self.qtd_creditos_ativos > self.qtd_creditos:
            raise ValueError("qtd_creditos_ativos não pode ser maior que qtd_creditos")
        return self

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )


class PredictResponse(BaseModel):
    score_risco: float | None = Field(
        description=(
            "Score de risco do modelo (0 a 1). Ordena clientes por risco; NÃO é uma "
            "probabilidade calibrada. Nulo quando uma regra de negócio decide sozinha."
        )
    )
    nivel_risco: Literal["baixo", "medio", "alto"]
    regra_aplicada: str | None = Field(
        default=None, description="Regra de negócio que definiu o risco, se houver"
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "indisponivel"]
    modelo_carregado: bool
    faixas: dict[str, float] | None = None
