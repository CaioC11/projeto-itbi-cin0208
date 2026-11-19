# Relatório de andamento — Predição do valor de imóveis com kNN

**Disciplina:** Ciência de Dados — CIN0208  
**Data:** 19/11/2026

## 1. Objetivo

O projeto investiga a previsão do valor de avaliação de imóveis do Recife a partir de dados do ITBI de 2025. O método principal é o k-Nearest Neighbors (kNN), com ênfase no efeito do tratamento de dados, da codificação de variáveis categóricas, da seleção de atributos, da normalização e da transformação da variável-alvo.

## 2. Análise exploratória

Foram realizadas análises da variável-alvo, das variáveis numéricas e das variáveis categóricas. A base contém 16.318 registros e 22 colunas. A variável `area_construida` apresentou correlação de Pearson próxima de 0,92 com o valor de avaliação. A distribuição do alvo é assimétrica à direita, com média superior à mediana e presença de valores muito altos. A transformação logarítmica reduziu a assimetria observada e foi incorporada ao protocolo de comparação.

A análise das distâncias mostrou que a escala das variáveis influencia fortemente o kNN. Sem normalização, `sfh` domina a distância euclidiana, enquanto a contribuição de `area_construida` fica subrepresentada. Esse resultado motivou a comparação explícita entre pipelines com e sem padronização.

## 3. Fundamentação conceitual

O kNN é um método de regressão baseado na proximidade entre observações no espaço de atributos. Para uma nova observação, o valor previsto é obtido a partir dos seus vizinhos mais próximos. Por depender diretamente de distâncias, o método é sensível à escala das variáveis e ao tratamento de valores ausentes. A validação cruzada é utilizada para estimar o desempenho em observações não utilizadas no ajuste, enquanto MAE, RMSE, MAPE e R² medem diferentes aspectos do erro de previsão.

## 4. Método

O pré-processamento é ajustado exclusivamente no conjunto de treino de cada partição. Valores zero em `area_terreno` e `fracao_ideal` são tratados como ausentes. As alternativas de tratamento de ausentes incluem remoção dos registros ou imputação pela mediana. As variáveis categóricas podem ser codificadas por One-Hot, Ordinal, Frequency ou Target Encoding.

O modelo principal é um `KNeighborsRegressor` com cinco vizinhos, distância euclidiana e pesos uniformes. A variável-alvo é transformada por `log1p` durante o treinamento e convertida novamente para a escala monetária na avaliação. A validação utiliza `KFold` com 10 partições, embaralhamento e `random_state=42`. As métricas são MAE, RMSE, MAPE e R².

O plano experimental prevê uma grade de 480 combinações, formada pela combinação de estratégias de tratamento de ausentes, codificação, escala, seleção de atributos, PCA, tratamento de `sfh` e opções de reamostragem. A seleção de atributos é calculada dentro do treino de cada fold para evitar vazamento de informação.

## 5. Resultados preliminares

Foi comparado um preditor baseado na média com duas versões do kNN, todas avaliadas por validação cruzada e alvo em escala logarítmica.

| Modelo | MAE médio | RMSE médio | MAPE médio | R² médio |
|---|---:|---:|---:|---:|
| Média em log | R$ 414.136 | R$ 2.300.199 | 0,747 | -0,016 |
| kNN, sem escala | R$ 200.656 | R$ 1.540.954 | 0,277 | 0,556 |
| kNN, com escala | **R$ 173.684** | **R$ 1.063.566** | **0,251** | **0,741** |

O kNN padronizado apresentou o melhor desempenho preliminar em todas as métricas comparáveis. Em relação ao kNN sem escala, a padronização reduziu o MAE médio em aproximadamente 13,5% e o RMSE médio em aproximadamente 31,0%. Esses resultados correspondem a uma configuração de referência, não à grade experimental completa.

## 6. Limitações e decisões pendentes

Ainda é necessário decidir se os registros com coordenadas ausentes devem ser removidos ou imputados, e se a imputação deve usar a mediana global ou a mediana por bairro. Também deve ser justificada a transformação logarítmica do alvo e avaliada a abordagem de regressão desbalanceada.

O uso de validação aleatória pode colocar no treino e no teste imóveis do mesmo prédio, produzindo dependência entre partições. Um particionamento por grupo, usando o prédio como unidade, é uma possibilidade de análise futura.

## 7. Próximas etapas

As próximas etapas são executar a grade completa de configurações, consolidar as métricas e comparar o desempenho por faixa de valor dos imóveis. Em seguida serão avaliados o tratamento de outliers, o alvo sem transformação, diferentes valores de `k`, métricas de distância e a reamostragem SMOGN. Os resultados serão usados na redação final e na preparação da apresentação.

## 8. Ambiente e referências

Os experimentos foram preparados em Python 3.12.15, com as bibliotecas especificadas em `requirements.txt`, principalmente pandas, NumPy e scikit-learn. O protocolo segue a documentação do scikit-learn para `KNeighborsRegressor`, `Pipeline`, `KFold` e `TransformedTargetRegressor`, além do material da disciplina sobre validação e avaliação de modelos de regressão.
