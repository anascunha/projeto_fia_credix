import joblib
import pandas as pd

# Carrega o modelo treinado
modelo = joblib.load("models/best_model.pkl")

# Carrega um cliente para teste
df = pd.read_csv("Dados/abt.csv")

# Seleciona um cliente
cliente = df.iloc[[0]]

# Remove colunas que não entram no modelo
cliente = cliente.drop(
    columns=[
        "TARGET",
        "SK_ID_CURR"
    ]
)

# Calcula a probabilidade de inadimplência
probabilidade = float(modelo.predict_proba(cliente)[0][1])


#Esses limites são apenas uma primeira versão. 
# No futuro cada cliente da Credix configure seus próprios intervalos 
# sem precisar alterar o modelo de Machine Learning
# Define a faixa de risco
if probabilidade < 0.20:
    risco = "baixo"
elif probabilidade < 0.50:
    risco = "medio"
else:
    risco = "alto"

# Resposta da API
resposta = {
    "probabilidade_inadimplencia": round(probabilidade, 4),
    "nivel_risco": risco
}

print(resposta)