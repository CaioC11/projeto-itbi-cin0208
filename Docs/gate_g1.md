# Gate G1 — Congelamento do Estágio 1

**Data:** 24/11/2026  
**Protocolo:** `KFold(n_splits=10, shuffle=True, random_state=42)`, alvo em `log1p`, avaliação na escala original e métricas por quartis do alvo e P99.

## Escopo concluído

Foram reexecutadas no ambiente virtual `.venv-validacao` as frentes F1-2, F1-4, F2-1, F2-2, F3-1, F3-2, F3-3 e F3-4. As linhas atualizadas passaram a registrar MAE, RMSE, MAPE, R², quartis Q1–Q4, P99, número de colunas e tempos de execução.

A frente F4-3 já possui os quatro métodos medidos em `resultados/experimentos_f4.csv`. SMOGN foi mantido fora da comparação principal por custo computacional. O arquivo `resultados/tempo_reamostragem.txt` registra os tempos por fold e deixa disponível a referência para o cache de reamostragem F4-4.

## Finalistas congelados

Os finalistas abaixo foram selecionados pelo menor MAE médio entre as configurações sem desbalanceamento, reunindo os resultados das frentes F1–F3. As técnicas de desbalanceamento ficam reservadas ao Estágio 3.

| Posição | ID | Encoding | Escala | Seleção | PCA | Ponderação | MAE médio | R² médio |
|---:|---|---|---|---|---|---|---:|---:|
| 1 | `535a463e31e8` | One-Hot | Standard | Nenhuma | Nenhum | Spearman | R$ 164.953 | 0,719 |
| 2 | `7b49d8ed9d18` | One-Hot | Robust | Nenhuma | Nenhum | Nenhuma | R$ 165.782 | 0,723 |
| 3 | `9552519b5450` | One-Hot | Standard | Nenhuma | Nenhum | MI | R$ 166.094 | 0,719 |
| 4 | `66a46645a28c` | One-Hot | Log1p + Z-score | Nenhuma | Nenhum | Nenhuma | R$ 168.850 | 0,638 |
| 5 | `1164ff41a8bc` | One-Hot | Standard | Spearman | Nenhum | Nenhuma | R$ 171.946 | 0,705 |

## Artefatos para a Etapa 2

- `resultados/experimentos.csv` contém os resultados atualizados de F1-2, F1-4, F2-1 e F2-2.
- `resultados/experimentos_f3.csv` contém os resultados atualizados de F3-1 a F3-4.
- `resultados/experimentos_f4.csv` contém os quatro métodos de F4-3 com métricas por faixa.
- `resultados/tempo_reamostragem.txt` registra o custo de geração das amostras e serve de base para F4-4.

O Gate G1 está pronto para a geração da grade combinada F5-5 usando os cinco IDs finalistas.
