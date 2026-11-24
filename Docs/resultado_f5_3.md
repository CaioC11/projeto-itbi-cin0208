# F5-3 — Baseline e Experimentos de Referência

**Projeto:** Predição de Valor de Imóveis com kNN — ITBI Recife  
**Disciplina:** Ciência de Dados (CIN0208) — UFPE  
**Responsável:** Pessoa B — F5 (Pipeline e Avaliação)

## 1. Objetivo

Reproduzir o baseline B0 utilizando o executor de experimentos desenvolvido na F5-1 e avaliar duas configurações adicionais:

- B0 com transformação logarítmica do alvo.
- B0 sem transformação logarítmica do alvo.
- B0 utilizando apenas registros sem valores ausentes.

O objetivo é estabelecer referências para comparar os diferentes métodos de pré-processamento utilizados nas demais etapas do projeto.

## 2. Metodologia

Todos os experimentos utilizaram:

- **Algoritmo:** kNN Regressor, com 5 vizinhos.
- **Distância:** Euclidiana.
- **Pesos:** Uniformes.
- **Validação cruzada:** 10 folds, com `shuffle=True` e `random_state=42`.
- **Imputação:** Mediana global dos atributos numéricos.
- **Encoding:** One-Hot Encoding.
- **Escalonamento:** StandardScaler.
- **Avaliação:** MAE, MAPE, RMSE e R².

O pré-processamento foi ajustado exclusivamente nos dados de treinamento de cada fold, evitando vazamento de informações do conjunto de teste.

Nos experimentos com transformação logarítmica, foi aplicada `log1p` ao alvo durante o treinamento e `expm1` às previsões para recuperar os valores em reais.

## 3. Resultados

| Métrica | B0 com log | B0 sem log | B0 sem faltantes |
|---|---:|---:|---:|
| MAE (R$) | 173.180,47 | 171.665,30 | 221.776,65 |
| RMSE (R$) | 1.062.885,08 | 993.462,92 | 1.203.669,62 |
| MAPE | 0,2522 | 0,2820 | 0,3045 |
