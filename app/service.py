import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

modelo = joblib.load(
    BASE_DIR / "models" / "best_model.pkl"
)

print("Modelo carregado com sucesso!")


def validar_regras(request):

    if request.renda_anual <= 0:
        return {
            "probabilidade_inadimplencia": 1.0,
            "nivel_risco": "alto"
        }

    comprometimento = (
        request.valor_parcela /
        request.renda_anual
    )

    # Parcela maior que a renda
    if comprometimento >= 1:
        return {
            "probabilidade_inadimplencia": 1.0,
            "nivel_risco": "alto"
        }

    return None


def prever(request):

    resposta = validar_regras(request)

    if resposta is not None:
        return resposta

    credito_renda = (
        request.valor_credito /
        request.renda_anual
    )

    parcela_renda = (
        request.valor_parcela /
        request.renda_anual
    )

    bem_renda = (
        request.valor_bem /
        request.renda_anual
    )

    renda_livre = (
        request.renda_anual -
        request.valor_parcela
    )

    comprometimento_renda = parcela_renda

    parcela_maior_renda = int(
        request.valor_parcela >
        request.renda_anual
    )

    credito_alto = int(
        credito_renda > 5
    )

    prazo_estimado = (
        request.valor_credito /
        request.valor_parcela
        if request.valor_parcela > 0
        else 0
    )

    debt_ratio = (
        request.valor_total_divida /
        request.valor_total_credito
        if request.valor_total_credito > 0
        else 0
    )

    entrada = pd.DataFrame([{

        "AMT_INCOME_TOTAL": request.renda_anual,
        "AMT_CREDIT": request.valor_credito,
        "AMT_ANNUITY": request.valor_parcela,
        "AMT_GOODS_PRICE": request.valor_bem,

        "NAME_EDUCATION_TYPE": request.escolaridade,
        "NAME_FAMILY_STATUS": request.estado_civil,

        "IDADE": request.idade,
        "TEMPO_EMPRESA": request.tempo_empresa,

        "CREDITO_RENDA": credito_renda,
        "PARCELA_RENDA": parcela_renda,
        "BEM_RENDA": bem_renda,

        "RENDA_LIVRE": renda_livre,
        "COMPROMETIMENTO_RENDA": comprometimento_renda,
        "PARCELA_MAIOR_RENDA": parcela_maior_renda,
        "CREDITO_ALTO": credito_alto,

        "PRAZO_ESTIMADO": prazo_estimado,

        "BUREAU_QTD_CREDITOS": request.qtd_creditos,
        "BUREAU_QTD_ATIVOS": request.qtd_creditos_ativos,
        "BUREAU_TOTAL_CREDITO": request.valor_total_credito,
        "BUREAU_TOTAL_DIVIDA": request.valor_total_divida,
        "BUREAU_DEBT_RATIO": debt_ratio

    }])

    probabilidade = float(
        modelo.predict_proba(entrada)[0][1]
    )

    if probabilidade < 0.20:
        risco = "baixo"
    elif probabilidade < 0.50:
        risco = "medio"
    else:
        risco = "alto"

    return {

        "probabilidade_inadimplencia": round(
            probabilidade,
            4
        ),

        "nivel_risco": risco

    }