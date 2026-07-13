import pandas as pd

from utils import *

#histórico de crédito

bureau = pd.read_csv(BUREAU)

qtd_creditos = (bureau.groupby("SK_ID_CURR")
.size()
.rename("BUREAU_QTD_CREDITOS")
)

ativos = (bureau["CREDIT_ACTIVE"] =="Active")
bureau["ATIVO"]=ativos.astype(int)
qtd_ativos=(
bureau.groupby("SK_ID_CURR")
["ATIVO"]
.sum()
.rename("BUREAU_QTD_ATIVOS")
)

credito_total=(
bureau
.groupby("SK_ID_CURR")
["AMT_CREDIT_SUM"]
.sum()
.rename("BUREAU_TOTAL_CREDITO")
)

divida_total=(
bureau
.groupby("SK_ID_CURR")
["AMT_CREDIT_SUM_DEBT"]
.sum()
.rename("BUREAU_TOTAL_DIVIDA")
)

bureau_features = pd.concat(
    [
        qtd_creditos,
        qtd_ativos,
        credito_total,
        divida_total
    ],
    axis=1
).reset_index()

bureau_features["BUREAU_DEBT_RATIO"] = (
    bureau_features["BUREAU_TOTAL_DIVIDA"] /
    bureau_features["BUREAU_TOTAL_CREDITO"].replace(0, 1)
)


bureau_features.to_csv(BUREAU_FEATURES,index=False)