# Comparativo — repo Gabi × repo Credix (Anna)

Comparação real, medida rodando os dois pipelines de ponta a ponta (não é só leitura de
código). Objetivo: decidir qual base usar em cada etapa do projeto.

## Escopo de dados

| | **Repo Gabi** | **Repo Credix (Anna)** |
|---|---|---|
| Tabelas usadas | 2 (`application_train` + `bureau`) | 5 (+ `previous_application`, `credit_card_balance`, `POS_CASH_balance`, `installments_payments`) |
| Features no modelo | ~20 (engenharia manual) | 416 (agregação automática por tabela-filha) |
| Tamanho da ABT | **57 MB** | 555 MB completa / 161 MB amostra (90k) |

## Velocidade (medida)

| Etapa | Repo Gabi | Repo Credix |
|---|---|---|
| Pipeline de dados (bronze→silver→gold) | ~17s (3 scripts) | mais lento (agrega 5 tabelas) |
| Treino (busca de hiperparâmetro) | **7s** (sem busca, hiperparâmetros fixos) | 4min25s (GridSearchCV, 15 fits) |
| Treino (`--fast`, direto com melhor param já conhecido) | — | **19s** (mesmos números do GridSearchCV completo) |
| `evaluation.ipynb` (4 algoritmos) | 1min25s | — |

## Qualidade do modelo (medida)

| Modelo | Accuracy | Recall | F1 | AUC teste | AUC treino | Gap overfit |
|---|---|---|---|---|---|---|
| Logística (Gabi, 20 feat.) | 0,598 | 0,601 | 0,195 | 0,635 | 0,635 | 0,000 |
| Árvore (Gabi) | 0,662 | 0,557 | 0,210 | 0,652 | 0,691 | 0,038 |
| Random Forest (Gabi) | **0,914** | **0,036** | 0,063 | 0,673 | **1,000** | **0,327** (decorou) |
| **XGBoost (Gabi, modelo final)** | 0,649 | 0,660 | 0,233 | 0,712 | 0,734 | 0,023 |
| Logística (Credix, 416 feat.) | 0,696 | 0,664 | 0,256 | **0,748** | 0,780 | 0,032 |
| XGBoost (Credix, 416 feat.) | — | 0,654 | — | **0,759** | — | — |

**Leitura:** o Random Forest confirma a armadilha clássica de base desbalanceada (acurácia
alta, recall quase zero, overfit total). O ganho de AUC do repo Credix (~0,03–0,05 acima
do Gabi) vem do maior número de tabelas/features — mais sinal disponível, ao custo de mais
peso computacional.

## Rigor e conformidade com a estrutura exigida

| Item | Repo Gabi | Repo Credix |
|---|---|---|
| `exp_analysis.ipynb` executado | ✅ (adicionado) | ✅ |
| `evaluation.ipynb` com comparação de algoritmos | ✅ (adicionado) | ✅ |
| Checagem de overfit (treino × teste) | ✅ (no notebook novo) | ✅ (já no `train.py`) |
| SHAP (explicabilidade) | ❌ | ✅ |
| GridSearchCV / tuning real documentado | ❌ (hiperparâmetros fixos) | ✅ |
| `requirements.txt` na raiz | ✅ (adicionado) | ✅ |
| Orquestração Airflow real | ✅ | ❌ (script Python substituto) |
| API estruturada (schemas/service) | ✅ | parcial |
| Regras de negócio explícitas na API | ✅ (ex.: parcela ≥ renda → risco alto sem chamar modelo) | ❌ |

## Decisão adotada

- **Entrega em grupo (13/07):** base **Gabi**, simplificada e rápida de rodar — já com os
  notebooks obrigatórios e `requirements.txt` adicionados.
- **Etapa individual (15/07, Anna):** base **Credix**, mais rigorosa (SHAP, tuning
  documentado, mais sinal/AUC), com modo `--fast` (19s) para não perder tempo ao vivo com
  GridSearchCV. Reaproveita a infraestrutura de deploy do Gabi (Airflow + camada de API com
  regras de negócio) adaptada para o modelo/pipeline do Credix.
