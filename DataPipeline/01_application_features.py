import pandas as pd

from utils import *

df = pd.read_csv(APPLICATION)

colunas = [
"SK_ID_CURR",
"TARGET",
"AMT_INCOME_TOTAL", #Renda anual do cliente
"AMT_CREDIT", #Valor do crédito solicitado
"AMT_ANNUITY",
"AMT_GOODS_PRICE",
"DAYS_BIRTH",
"DAYS_EMPLOYED",
# "CODE_GENDER",
# "FLAG_OWN_CAR",
# "FLAG_OWN_REALTY",
"NAME_EDUCATION_TYPE",
"NAME_FAMILY_STATUS"
]

df = df[colunas]

df["IDADE"] = abs(df["DAYS_BIRTH"]) / 365

#tratar pq esse valor é absurdo 
df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, pd.NA)
df["TEMPO_EMPRESA"] = abs(df["DAYS_EMPLOYED"]) / 365
df["TEMPO_EMPRESA"] = df["TEMPO_EMPRESA"].fillna(0)

df.drop(
    columns=[
        "DAYS_BIRTH",
        "DAYS_EMPLOYED"
    ],
    inplace=True
)

renda = df["AMT_INCOME_TOTAL"].replace(0, pd.NA)

df["CREDITO_RENDA"] = df["AMT_CREDIT"] / renda
df["PARCELA_RENDA"] = df["AMT_ANNUITY"] / renda
df["BEM_RENDA"] = df["AMT_GOODS_PRICE"] / renda

df["RENDA_LIVRE"] = (
    df["AMT_INCOME_TOTAL"] -
    df["AMT_ANNUITY"]
)

df["COMPROMETIMENTO_RENDA"] = (
    df["AMT_ANNUITY"] /
    renda
).fillna(0)

df["PARCELA_MAIOR_RENDA"] = (
    df["AMT_ANNUITY"] >
    df["AMT_INCOME_TOTAL"]
).astype(int)

df["CREDITO_ALTO"] = (
    df["CREDITO_RENDA"] > 5
).astype(int)

df[[
    "CREDITO_RENDA",
    "PARCELA_RENDA",
    "BEM_RENDA"
]] = df[[
    "CREDITO_RENDA",
    "PARCELA_RENDA",
    "BEM_RENDA"
]].fillna(0)


# df["FLAG_OWN_CAR"] = (
#     df["FLAG_OWN_CAR"] == "Y"
# ).astype(int)

# df["FLAG_OWN_REALTY"] = (
#     df["FLAG_OWN_REALTY"] == "Y"
# ).astype(int)

df["PRAZO_ESTIMADO"] = (
    df["AMT_CREDIT"] /
    df["AMT_ANNUITY"].replace(0, pd.NA)
).fillna(0)

#tratar negativo
df["CREDITO_RENDA"] = df["CREDITO_RENDA"].clip(lower=0)
df["PARCELA_RENDA"] = df["PARCELA_RENDA"].clip(lower=0)
df["BEM_RENDA"] = df["BEM_RENDA"].clip(lower=0)

#arredonda valor
colunas_float = [
    "IDADE",
    "TEMPO_EMPRESA",
    "CREDITO_RENDA",
    "PARCELA_RENDA",
    "BEM_RENDA",
    "PRAZO_ESTIMADO"
]
df[colunas_float] = df[colunas_float].round(2)


assert df["SK_ID_CURR"].is_unique

df.to_csv(APPLICATION_FEATURES,index=False)
