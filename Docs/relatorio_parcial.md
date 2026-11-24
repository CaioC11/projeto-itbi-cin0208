# Relatório de andamento — Predição do valor de imóveis com kNN
## 1. Objetivo
## 2. Análise exploratória
## 3. Fundamentação conceitual
## 4. Método
O plano experimental prevê 37 experimentos no Estágio 1 + 144 no Estágio 2 = 181 combinações (≥ 166 sem duplicatas), com a F4 como bônus, cobrindo tratamento de ausentes, codificação, escala, seleção de atributos, PCA, tratamento de `sfh` e opções de reamostragem. A seleção de atributos é calculada dentro do treino de cada fold para evitar vazamento de informação.
## 5. Resultados preliminares
## 6. Limitações e decisões pendentes
## 7. Próximas etapas
## 8. Ambiente e referências
Os experimentos foram preparados em Python 3.12.15 (conferir padronização do grupo em relação a 3.12.10), com as bibliotecas especificadas em `requirements.txt`, principalmente pandas, NumPy e scikit-learn. O protocolo segue a documentação do scikit-learn para `KNeighborsRegressor`, `Pipeline`, `KFold` e `TransformedTargetRegressor`, além do material da disciplina sobre validação e avaliação de modelos de regressão.
