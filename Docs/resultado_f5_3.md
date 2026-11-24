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
| R² | 0,7413 | 0,7617 | 0,6317 |
| MAE em log | 0,1956 | 0,1930 | 0,2377 |

Os resultados correspondem às médias das métricas obtidas nos 10 folds.

### 3.1. Efeito da transformação logarítmica

O B0 sem transformação logarítmica apresentou menor MAE e RMSE, além de maior R².

Em contrapartida, o B0 com log apresentou menor MAPE, indicando melhor desempenho relativo em termos de erro percentual.

Portanto, a transformação logarítmica não apresentou superioridade em todas as métricas avaliadas.

### 3.2. Efeito da remoção de registros com valores ausentes

O experimento sem faltantes apresentou MAE de R$ 221.776,65 e R² de 0,6317.

Entretanto, esse experimento utiliza um subconjunto diferente do dataset original. Assim, a diferença observada não permite concluir isoladamente que a remoção de linhas prejudicou o desempenho do modelo.

Para avaliar adequadamente essa técnica, é necessário comparar métodos sobre o mesmo conjunto de observações.

## 4. Divergência do baseline versionado

Durante a validação da F5-1, foi identificada uma diferença entre os resultados armazenados originalmente no repositório e os obtidos na execução atual.

| Referência | MAE (R$) | R² |
|---|---:|---:|
| `baseline.csv` original do Git | 173.684,04 | 0,741 |
| `baseline.py` executado localmente | 173.180,47 | 0,741 |
| `executor.py` — F5-1 | 173.180,47 | 0,741 |

A diferença absoluta entre os valores de MAE é de aproximadamente **R$ 503,57 (0,29%)**.

Foram realizadas as seguintes verificações:

- O arquivo `src/baseline.py` não apresentava alterações locais em relação à versão registrada no Git.
- O dataset original também não apresentava alterações locais.
- As versões das bibliotecas estavam de acordo com o `requirements.txt`.
- O executor F5-1 reproduziu o resultado obtido pela execução atual do `baseline.py`.

**Versões utilizadas:**

| Biblioteca | Versão |
|---|---|
| Python | 3.12 |
| NumPy | 1.26.4 |
| Pandas | 2.2.2 |
| Scikit-learn | 1.5.1 |

A causa exata da divergência histórica não foi identificada. Portanto, o resultado original foi preservado como referência histórica, enquanto o valor de R$ 173.180,47 representa a reprodução obtida no ambiente atual.

## 5. Reprodutibilidade

Os experimentos foram executados utilizando a função `executar_experimento()` de `src/executor.py`.

Os resultados estão armazenados em:

`resultados/referencias_f5_3.csv`

O arquivo contém as configurações utilizadas, métricas globais, métricas por faixa de preço, tempos de execução e os MAEs individuais dos 10 folds.

A identificação de cada experimento é realizada por uma chave baseada na configuração, permitindo identificar experimentos repetidos.

## 6. Conclusão

A F5-3 permitiu reproduzir o baseline atual e estabelecer referências adicionais para os experimentos de pré-processamento.

O executor apresentou resultados equivalentes aos da execução atual do `baseline.py`, confirmando a consistência da avaliação implementada.

A comparação também demonstrou que a transformação logarítmica apresenta vantagens e desvantagens dependendo da métrica considerada.

A divergência em relação ao arquivo histórico do Git foi identificada e documentada, permanecendo sem causa confirmada.

Os resultados obtidos servirão como referência para a comparação dos experimentos das etapas seguintes.