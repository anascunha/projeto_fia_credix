# RUNBOOK — Credix (repo final)

Comandos exatos para rodar o projeto do zero, com o tempo esperado de cada etapa
(medido). Use isso para ensaiar antes da apresentação — se algum passo demorar muito
mais que o indicado, é sinal de problema (ambiente, dados errados), não normal.

## 0. Pré-requisitos

```bash
pip install -r requirements.txt
```

Coloque os CSVs da Home Credit em:

```text
Dados/raw_data/application_train.csv
Dados/raw_data/bureau.csv
```

(Baixados do Kaggle — Home Credit Default Risk. Não versionados no repo.)

## 1. Pipeline de dados

Rodar de dentro de `DataPipeline/`:

```bash
cd DataPipeline
python 01_application_features.py   # ~7s  -> ../Dados/clean_data/application_features.csv
python 02_bureau_features.py        # ~5s  -> ../Dados/clean_data/bureau_features.csv
python merge_abt.py                 # ~5s  -> ../Dados/abt.csv (~57MB, 307.511 linhas)
```

**Tempo total: ~17s.** Se travar ou demorar minutos, confira se os CSVs de entrada estão
no lugar certo e não estão corrompidos/truncados.

## 2. Notebooks (evidência documentada — não precisa rodar ao vivo)

Já vêm executados no repo. Para reexecutar do zero (opcional, útil para conferir):

```bash
cd DataPipeline
jupyter nbconvert --to notebook --execute --inplace exp_analysis.ipynb   # ~17s

cd ../Model
jupyter nbconvert --to notebook --execute --inplace evaluation.ipynb     # ~1min25s
```

`evaluation.ipynb` treina os 4 algoritmos (Logística, Árvore, Random Forest, XGBoost) —
é o mais demorado. Não precisa rodar isso ao vivo na apresentação: mostre os outputs já
salvos no notebook.

## 3. Treino do modelo final

Rodar de dentro de `Model/`:

```bash
cd Model
python train.py   # ~7s -> ../models/best_model.pkl + ../metrics/train_metrics.json
```

**Tempo esperado: ~7s.** Sem busca de hiperparâmetro (valores já fixos no `config.py` /
`train.py`) — por isso é rápido o suficiente para rodar ao vivo sem gerar silêncio longo
na apresentação.

Saída esperada (aprox.):

```text
===== XGBoost =====
accuracy: 0.6486
precision: 0.1412
recall: 0.6598
f1: 0.2326
roc_auc: 0.7118
```

Se os números vierem muito diferentes disso, o dado de entrada mudou — não é erro de
código.

## 4. API de predição (FastAPI)

```bash
cd .. # rodar na raiz do projeto
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

API carrega neste: http://localhost:8000/docs

Exemplo de request no `/predict` (ver README seção 10 para o payload completo).

## 5. Orquestração (Airflow) — NÃO subir ao vivo na apresentação

A stack sobe 7 containers (postgres, redis, api-server, scheduler, worker,
dag-processor, triggerer) e o próprio `airflow-init` exige no mínimo 4GB de RAM e 2 CPUs.
Cold start pode levar minutos.

**Suba com antecedência**, antes da apresentação/defesa começar:

```bash
docker compose up -d
```

Interface: http://localhost:8080 (usuário/senha padrão: `airflow`/`airflow`)

Na hora da apresentação, **só mostre a interface com o DAG já executado** — não dispare
nem suba a stack ao vivo. Se precisar rodar de novo, use o botão "Trigger DAG" na UI
(instantâneo, já com tudo no ar) em vez de `docker compose up`.

## Checklist rápido antes de apresentar

- [ ] `Dados/raw_data/*.csv` no lugar
- [ ] Pipeline de dados rodado sem erro (~17s)
- [ ] `models/best_model.pkl` existe e é recente
- [ ] Notebooks com output visível (não precisa reexecutar)
- [ ] Se for demonstrar API: `uvicorn` rodando, `/docs` acessível
- [ ] Se for demonstrar Airflow: stack já no ar **antes** de começar, DAG já com uma
      execução bem-sucedida visível na UI
