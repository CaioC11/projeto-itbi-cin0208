# Executor de Experimentos

O executor realiza experimentos de regressão utilizando **kNN (k=5)** e validação cruzada de **10 folds**.

## Metodologia

- Validação cruzada com `KFold`, utilizando `shuffle=True` e `random_state=42`.
- Transformação do alvo com `log1p` e retorno das previsões com `expm1`.
- Pré-processamento ajustado exclusivamente nos dados de treino de cada fold.
- Suporte a filtros de treino e reamostragem opcionais.
- Modelo kNN com distância euclidiana e 5 vizinhos.

## Métricas

São calculadas as seguintes métricas:

- MAE (Mean Absolute Error).
- MAPE (Mean Absolute Percentage Error).
- RMSE (Root Mean Squared Error).
- R² (Coeficiente de determinação).

Os resultados incluem média e desvio-padrão das métricas, além dos MAEs individuais de cada fold para análises estatísticas posteriores.

## Execução

A partir da raiz do repositório:

```bash
python src/executor.py
```

Para executar sem registrar resultados:

```bash
python src/executor.py --sem-salvar
```

Para executar uma parte da grade:

```bash
python src/executor.py --parte 1/4
```

A divisão permite distribuir os experimentos entre quatro máquinas, evitando sobreposição de configurações.

Também é possível definir outro arquivo de saída:

```bash
python src/executor.py --saida /tmp/teste_executor.csv
```

## Resultados

Por padrão, os resultados são registrados em:

```text
resultados/experimentos.csv
```

O CSV armazena as configurações utilizadas, métricas, tempos de execução e resultados individuais dos folds.

## Integração com a grade de experimentos

A função `executar_grade()` permite receber uma lista de configurações e executar somente a parte destinada a cada máquina.

A função `executar_experimento()` também permite fornecer filtros de treino, reamostragem e diferentes configurações de pré-processamento.

**Importante:** a execução pelo terminal atualmente utiliza uma configuração B0 de referência. A grade completa deve ser fornecida à função `executar_grade()`.