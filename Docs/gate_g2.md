# Gate G2 — Estágio 2 (F5-5)

**Data:** 25/11/2026  
**Protocolo:** grade combinada gerada por `itertools.product`, validação cruzada com 10 folds, `shuffle=True`, `random_state=42`, alvo em `log1p` e avaliação na escala original.

## Execução

A grade F5-5 contém 144 combinações únicas de tratamento de faltantes, outliers, encoding, escala e ponderação. Ela foi dividida em quatro partes de 36 configurações e executada com:

```text
python src/grade_f5.py --executar --confirmar-g1 --parte 1/4
python src/grade_f5.py --executar --confirmar-g1 --parte 2/4
python src/grade_f5.py --executar --confirmar-g1 --parte 3/4
python src/grade_f5.py --executar --confirmar-g1 --parte 4/4
```

Cada resultado contém MAE, RMSE, MAPE, R², tempos, dimensionalidade e métricas de erro por Q1–Q4 e P99. A consolidação está em `resultados/experimentos_f5.csv`.

Para a análise conjunta, foram deduplicados os resultados da F5-5 com os resultados oficiais do Estágio 1. O conjunto final contém 171 configurações únicas, acima da meta mínima de 150. As tabelas de ranking, custo, impacto das técnicas e Wilcoxon estão em `resultados/analise_f5/`.

## Top-5 congelado para o Gate G2

| Posição | ID | Faltantes | Encoding | Escala | Ponderação | MAE médio | R² médio |
|---:|---|---|---|---|---|---:|---:|
| 1 | `40ff390453f0` | Mediana por bairro | Ordinal | Robust | Nenhuma | R$ 159.938 | 0,771 |
| 2 | `b33277894c12` | Mediana por bairro | Ordinal | Standard | Spearman | R$ 160.059 | 0,751 |
| 3 | `5985b3967252` | Mediana global | Ordinal | Standard | Spearman | R$ 160.772 | 0,751 |
| 4 | `4ce55c93f720` | Média | Ordinal | Standard | Spearman | R$ 161.071 | 0,750 |
| 5 | `5f9ba9069249` | Mediana global | Ordinal | Robust | Nenhuma | R$ 161.868 | 0,771 |

Os cinco IDs também estão registrados em `resultados/top5_g2.csv`. Eles são a entrada do Estágio 3, que cruzará cada configuração com as técnicas de desbalanceamento usando o cache F4-4.
