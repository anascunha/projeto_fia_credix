# Como rodar o Credix

Passo a passo para executar o projeto do zero. Os tempos indicados foram medidos
pela equipe numa execução com a base completa (307.511 clientes); se uma etapa demorar
muito mais que isso, provavelmente há problema no ambiente ou nos dados de entrada.

## 0. Pré-requisitos

- Python 3.12 ou 3.13
- Docker + Docker Compose (só para a API em container e para o Airflow)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Dados

Os dados vêm da competição **Home Credit Default Risk** do Kaggle
(`home-credit-default-risk`) e não são versionados no repositório. Para baixar:

1. Crie uma conta no Kaggle e aceite as regras da competição em
   https://www.kaggle.com/competitions/home-credit-default-risk/rules
2. Autentique a API do Kaggle: `kaggle auth login` (pelo navegador) ou exporte
   `KAGGLE_API_TOKEN` com o token gerado em https://www.kaggle.com/settings/api
3. Rode:

```bash
python DataPipeline/00_baixar_dados.py
```

O script baixa só os dois arquivos usados pelo projeto e pula os que já existem:

```text
Dados/raw_data/application_train.csv
Dados/raw_data/bureau.csv
```

Se preferir, baixe manualmente pela página da competição e coloque os CSVs nesses caminhos.

## 1. Testes (não precisam dos dados do Kaggle)

```bash
pytest
```

Os testes geram dados **sintéticos** no formato dos CSVs da Home Credit e cobrem as
features, as regras de negócio, o treino e a API.

## 2. Pipeline completo (features → ABT → treino)

```bash
python pipeline_orchestration.py
```

Ou etapa por etapa (podem ser chamados de qualquer pasta):

```bash
python DataPipeline/01_application_features.py   # ~7s -> Dados/clean_data/application_features.csv
python DataPipeline/02_bureau_features.py        # ~5s -> Dados/clean_data/bureau_features.csv
python DataPipeline/merge_abt.py                 # ~5s -> Dados/abt.csv (~57MB, 307.511 linhas)
python Model/train.py                            # ~7s -> models/ e metrics/
```

O treino gera:

| Arquivo | Conteúdo |
|---|---|
| `models/best_model.pkl` | pipeline Scikit-learn (One-Hot + XGBoost) |
| `models/faixas_risco.json` | cortes de score das faixas baixo/médio/alto |
| `metrics/train_metrics.json` | métricas no conjunto de teste e inadimplência por faixa |
| `metrics/importancia_features.csv` | importância das features no XGBoost |

Saída de referência das métricas (execução da equipe com a base completa):

```text
accuracy: 0.6486
precision: 0.1412
recall: 0.6598
f1: 0.2326
roc_auc: 0.7118
```

Diferenças grandes em relação a esses números indicam que o dado de entrada mudou.

## 3. Notebooks

Já vêm executados no repositório (outputs salvos). Para reexecutar:

```bash
jupyter nbconvert --to notebook --execute --inplace DataPipeline/exp_analysis.ipynb   # ~17s
jupyter nbconvert --to notebook --execute --inplace Model/evaluation.ipynb           # ~1min25s
```

`evaluation.ipynb` treina 4 algoritmos (Logística, Árvore, Random Forest, XGBoost) e gera o
gráfico de importância de features em `docs/img/importancia_features.png`.

## 4. API de predição

Local:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Ou em container (precisa de `models/` gerado no passo 2):

```bash
docker compose up -d --build credix-api
```

- Documentação interativa: http://localhost:8000/docs
- Saúde: `curl http://localhost:8000/health`
- Predição:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"idade": 42, "escolaridade": "Higher education", "estado_civil": "Married",
       "renda_anual": 250000, "tempo_empresa": 12, "valor_credito": 180000,
       "valor_bem": 220000, "valor_parcela_anual": 42000, "qtd_creditos": 4,
       "qtd_creditos_ativos": 2, "valor_total_credito": 600000, "valor_total_divida": 50000}'
```

## 5. Orquestração com Airflow

```bash
cp .env.example .env    # preencha FERNET_KEY e AIRFLOW_UID conforme as instruções do arquivo
docker compose up -d
```

A stack sobe 7 containers (postgres, redis, api-server, scheduler, worker, dag-processor,
triggerer). O `airflow-init` pede pelo menos 4 GB de RAM e 2 CPUs, e a primeira subida
pode levar alguns minutos.

- Interface: http://localhost:8080 (usuário/senha padrão: `airflow`/`airflow`)
- Ative e dispare a DAG `credix_pipeline`. Ela tem uma tarefa por etapa:
  `application_features` e `bureau_features` (em paralelo) → `construir_abt` → `treinar_modelo`.

## Checklist

- [ ] `Dados/raw_data/*.csv` no lugar
- [ ] `pytest` passando
- [ ] `models/best_model.pkl` e `models/faixas_risco.json` gerados
- [ ] API respondendo em `/health` com `"status": "ok"`
