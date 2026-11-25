# Gate G3 - Estagio 3 (F4-5: Desbalanceamento no Top-5)

**Data:** 25/11/2026
**Protocolo:** cruzamento das 5 configuracoes congeladas no Gate G2 (`resultados/top5_g2.csv`) com as 4 tecnicas de reamostragem aprovadas no Gate G1 (`random_over`, `random_under`, `gaussian_noise`, `smoter`), aplicadas exclusivamente no treino de cada fold (`KFold(10, shuffle=True, random_state=42)`), alvo em `log1p` e avaliacao na escala original.

## Execucao

Os 20 experimentos (`5 pipelines x 4 tecnicas`) foram executados via `python src/rodar_f4_5.py` e salvos em `resultados/experimentos_f4_5.csv`. O metodo `SMOGN` foi mantido fora da grade conforme decisao do Gate G1 devido ao custo computacional inviavel.

## Resultados consolidados (Top-5 do Estagio 3)

| Posicao | ID | Faltantes | Escala | Ponderacao | Desbalanceamento | MAE medio | RMSE medio | R2 medio | Q4 MAE | P99 MAE |
|---:|---|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | `daafa462a0a2` | mediana_bairro | standard | pond_spearman | `random_over` | R$ 149,804.25 | R$ 805,462.48 | 0.8279 | R$ 462,394.49 | R$ 5,170,133.72 |
| 2 | `4ca6eef60157` | mediana_bairro | standard | pond_spearman | `gaussian_noise` | R$ 150,067.12 | R$ 799,185.82 | 0.8326 | R$ 464,977.25 | R$ 5,148,342.16 |
| 3 | `6cac2321c245` | mediana_global | standard | pond_spearman | `random_over` | R$ 150,072.66 | R$ 803,842.66 | 0.8300 | R$ 464,299.42 | R$ 5,149,821.86 |
| 4 | `2ac28a188d83` | mediana_bairro | standard | pond_spearman | `smoter` | R$ 150,180.26 | R$ 842,655.10 | 0.8038 | R$ 464,113.68 | R$ 5,226,199.10 |
| 5 | `9afc82770c03` | media | standard | pond_spearman | `random_over` | R$ 150,201.81 | R$ 803,708.74 | 0.8298 | R$ 464,249.72 | R$ 5,149,348.95 |

## Selecao do modelo campeao (Gate G3)

- **Campeao oficial (`daafa462a0a2`):** `mediana_bairro` + `ordinal` + `standard` + `pond_spearman` + `random_over`.
- **Desempenho global:** MAE = **R$ 149,804.25** (-R$ 10,134.22 vs melhor G2; -R$ 23,376.22 vs baseline B0) e R2 = **0.8279**.
- **Cauda direita (imoveis caros):** Q4 MAE caiu de R$ 503,223.85 para **R$ 462,394.49** (-R$ 40,829.36) e P99 MAE caiu de R$ 6,048,073.09 para **R$ 5,170,133.72** (-R$ 877,939.37).
- **Vice-campeao (`4ca6eef60157` - `gaussian_noise`):** MAE = R$ 150,067.12, maior R2 (**0.8326**) e menor P99 MAE (**R$ 5,148,342.16**).
- **Descarte de `random_under`:** degradou o MAE em todos os cenarios (R$ 166,9 mil a R$ 169,6 mil).

## Encaminhamento para Etapa 4 (F5-6)

- Incluir `resultados/experimentos_f4_5.csv` em `src/analise_f5.py` para consolidar os 202 experimentos unicos, rodar o teste de Wilcoxon por fold e congelar as tabelas e graficos finais.