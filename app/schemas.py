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
    estar na MESMA periodicidade (mensal), como no treino do modelo.
    """

    id_cliente: str | None = Field(
        default=None,
        max_length=100,
        description="Identificador do cliente no sistema de quem chama (opcional, só para o registro)",
    )
    idade: float = Field(ge=18, le=100, description="Idade em anos")
    escolaridade: Escolaridade
    estado_civil: EstadoCivil
    renda_mensal: float = Field(ge=0, description="Renda mensal (AMT_INCOME_TOTAL)")
    tempo_empresa: float = Field(ge=0, description="Tempo no emprego atual, em anos")
    valor_credito: float = Field(ge=0, description="Valor do crédito solicitado (AMT_CREDIT)")
    valor_bem: float = Field(ge=0, description="Valor do bem financiado (AMT_GOODS_PRICE)")
    valor_parcela_mensal: float = Field(
        ge=0, description="Valor mensal da parcela do empréstimo (AMT_ANNUITY)"
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
        }
    )


class PredictResponse(BaseModel):
    id_solicitacao: str | None = Field(
        description="Identificador da previsão no registro (nulo se o registro estiver indisponível)"
    )
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
    versao_modelo: str | None = None
    registro_ativo: bool = False
    faixas: dict[str, float] | None = None


class Predicao(BaseModel):
    """Uma previsão registrada, com a entrada, as features e o resultado."""

    id_solicitacao: str
    id_cliente: str | None
    data_previsao: str
    versao_modelo: str | None
    entrada: dict
    features: dict
    score_risco: float | None
    nivel_risco: Literal["baixo", "medio", "alto"]
    regra_aplicada: str | None
    resultado_observado: bool | None = Field(
        description="True = inadimplente, False = adimplente, nulo = ainda em maturação"
    )
    data_resultado: str | None


class ResultadoObservado(BaseModel):
    inadimplente: bool = Field(description="O cliente ficou inadimplente no período de maturação?")
