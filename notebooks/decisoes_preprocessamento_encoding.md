# Registro de Decisões — Pré-tratamento e Encoding (`F1-5` / Issue #40)

Este documento consolida os experimentos conduzidos em validação cruzada *10-fold* (`seed=42`) sobre o dataset do ITBI (16.016 instâncias após a limpeza fixa em `src/limpeza.py`) utilizando o regressor de referência (`KNeighborsRegressor(k=5)` com alvo transformado via $\log(1 + y)$), bem como as decisões técnicas adotadas para as próximas etapas de modelagem.

---

## 1. Configuração de Referência (`Baseline B0`)

Para isolar o impacto de cada decisão experimental, definiu-se a configuração base **`B0`** variando um único fator por vez:
* **Conjunto:** `completo` (16.016 instâncias)
* **Faltantes:** `mediana_global` (`SimpleImputer(strategy="median")`)
* **Outliers:** `nenhum`
* **Variável `sfh`:** `original`
* **Encoding:** `OrdinalOrdenado` em `padrao_acabamento` e `OneHotEncoder` nas demais categóricas
* **Escala:** `StandardScaler` aplicado após imputação e codificação

| Configuração | Colunas | MAE (R$) | MAPE (%) | RMSE (R$) | $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`B0 (completo)`** | 131,7 | 200.053,49 | 27,16% | 1.491.937 | 0,5668 |
| **`B0 (sem_faltantes)`** | 115,5 | 243.210,65 | 32,50% | 1.320.876 | 0,6142 |

---

## 2. Resultados Experimentais por Frente

### 2.1. Tratamento de Valores Faltantes (`F1-2` / Issue #37)

| Estratégia | Colunas | MAE (R$) | MAPE (%) | RMSE (R$) | $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `remover_linhas` (`B0 sem_faltantes`) | 115,5 | 243.210,65 | 32,50% | 1.320.876 | 0,6142 |
| `sem_lat_lon` (descarta coordenadas + mediana) | 129,7 | **195.754,35** | 26,85% | 1.485.286 | 0,5715 |
| **`mediana_bairro`** (mediana por bairro + fallback global) | 131,7 | 197.903,58 | **26,82%** | **1.480.660** | **0,5770** |
| `mediana_bairro_indicador` (+ colunas `_missing`) | 135,6 | 199.289,60 | 27,01% | 1.480.747 | 0,5769 |
| `area_terreno_tipo` (mediana por `tipo_imovel`) | 131,7 | 200.046,80 | 27,15% | 1.491.937 | 0,5668 |
| `media` (`SimpleImputer(strategy="mean")`) | 131,7 | 200.233,31 | 27,10% | 1.491.856 | 0,5670 |

* **Análise:** A remoção de linhas (`remover_linhas`) descarta ~43% dos registros e degrada severamente o MAE (+R$ 43.157 em relação ao `B0`). Entre as imputações, **`mediana_bairro`** obteve os melhores valores de $R^2$ (`0,5770`), RMSE (`1.480.660`) e MAPE (`26,82%`), preservando a estrutura espacial local e mantendo `latitude`/`longitude` disponíveis para modelos baseados em árvores.
* **Decisão:** Adotar **`mediana_bairro`** como estratégia padrão de imputação.

---

### 2.2. Tratamento de Outliers (`F1-3` / Issue #38)

*(Aplicado exclusivamente sobre os dados de treino dentro de cada fold, mantendo a validação intacta.)*

| Estratégia | Colunas | MAE (R$) | MAPE (%) | RMSE (R$) | $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`nenhum` (`B0`)** | 131,7 | **200.053,49** | 27,16% | **1.491.937** | **0,5668** |
| `winsor_1_99` (percentis 1% e 99%) | 131,7 | 207.744,10 | 25,38% | 1.862.040 | 0,3657 |
| `winsor_5_95` (percentis 5% e 95%) | 131,7 | 214.192,00 | 25,78% | 1.875.331 | 0,3561 |
| `iqr_log_3.0` (remoção no treino via $3,0 \times \text{IQR}$) | 131,7 | 217.228,73 | **24,81%** | 2.077.611 | 0,2082 |
| `iqr_log_1.5` (remoção no treino via $1,5 \times \text{IQR}$) | 129,2 | 248.520,19 | 28,48% | 2.216.887 | 0,0734 |

* **Análise:** Após a limpeza fixa (que já remove o registro anômalo de R$ 19,52 e zeros inválidos), os valores extremos restantes representam imóveis reais de alto padrão ou grandes empreendimentos comerciais. Cortar (`IQR`) ou truncar (`Winsorização`) esses registros faz com que o modelo subestime sistematicamente imóveis de alto valor, derrubando o $R^2$ de `0,5668` para a faixa de `0,0734–0,3657` e elevando o RMSE em até +48%.
* **Decisão:** Manter **`nenhum`** tratamento adicional de outliers.

---

### 2.3. Tratamento da Variável `sfh` (Issue #39)

| Estratégia | Colunas | MAE (R$) | MAPE (%) | RMSE (R$) | $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`remover`** (descartar `sfh`) | 130,7 | **195.100,63** | **26,67%** | **1.490.763** | **0,5676** |
| `original` (`B0`) | 131,7 | 200.053,49 | 27,16% | 1.491.937 | 0,5668 |
| `binarizar` (`sfh > 0`) | 131,7 | 200.485,09 | 27,70% | 1.491.897 | 0,5668 |

* **Análise:** A remoção da coluna `sfh` reduziu o MAE em **R$ 4.952,86** e o MAPE em `0,49 p.p.`, indicando que a variável adiciona ruído nas métricas de distância sem ganho preditivo.
* **Decisão:** Adotar **`remover`** para a coluna `sfh`.

---

### 2.4. Encoding de Variáveis Categóricas (`F2-1` e `F2-2` / Issue #42)

| Estratégia | Colunas | MAE (R$) | MAPE (%) | RMSE (R$) | $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`target_forte_100`** (`suavizacao=100` em alta card. + OHE baixa card.) | **17,0** | **172.919,41** | **25,41%** | **1.002.346** | **0,7754** |
| `frequency` (frequência relativa em alta card. + OHE baixa card.) | 17,0 | 173.260,12 | 25,51% | 1.036.030 | 0,7557 |
| `target_suave_10` (`suavizacao=10` em alta card. + OHE baixa card.) | 17,0 | 175.574,41 | 25,41% | 1.044.874 | 0,7413 |
| `hibrido_te_bairro_oh_resto` (TE em `bairro` + OHE nas demais) | 48,8 | 190.722,69 | 25,75% | 1.458.232 | 0,5847 |
| `onehot_sem_bairro` (OHE sem a coluna `bairro`) | 47,8 | 191.079,85 | 25,76% | 1.460.312 | 0,5830 |
| `onehot_bairro_raros_30` (OHE agrupando bairros com $<30$ registros) | 99,1 | 199.407,56 | 26,86% | 1.491.697 | 0,5670 |
| `ordinal` (`padrao_acabamento` e `estado_conservacao` ordinais + OHE) | 129,7 | 199.588,13 | 27,13% | 1.490.894 | 0,5675 |
| `onehot_peso_0.5` (OHE com peso 0,5) | 131,7 | 200.053,49 | 27,16% | 1.491.937 | 0,5668 |
| `onehot` (`B0`) | 131,7 | 200.053,49 | 27,16% | 1.491.937 | 0,5668 |

* **Análise:** O *One-Hot Encoding* expande o espaço para ~132 dimensões esparsas, prejudicando o cálculo de distância euclidiana no kNN. Ao substituir as colunas de alta cardinalidade (`bairro`, `tipo_imovel`, `tipo_construcao`) por **`target_forte_100`**, a dimensionalidade cai para **17 colunas** e o desempenho salta expressivamente: redução de **13,6% no MAE** (-R$ 27.134) e aumento de **+0,2086 no $R^2$** (`0,5668` $\rightarrow$ `0,7754`). A suavização mais forte (`100` vs. `10`) também previne *overfitting* em bairros ou tipos de construção com poucas amostras.
* **Decisão:** Adotar **`target_forte_100`** (`OrdinalOrdenado` em `padrao_acabamento`, `TargetEncodingSuave(suavizacao=100)` em `bairro`, `tipo_imovel` e `tipo_construcao`, e `OneHotRaras` em `tipo_ocupacao` e `estado_conservacao`).

---

## 3. Resumo da Configuração Consolidada

| Etapa | Decisão Adotada | Justificativa Principal |
| :--- | :--- | :--- |
| **Limpeza Fixa** | `src/limpeza.py` | Remove duplicatas exatas, `valor_avaliacao <= 100`, zeros estruturais e padroniza grafia de `bairro` (84 bairros únicos). |
| **Faltantes (`F1-2`)** | `mediana_bairro` | Melhor $R^2$ (`0,5770`), RMSE e MAPE sem perda de registros nem descarte de coordenadas geográficas. |
| **Outliers (`F1-3`)** | `nenhum` | Preserva imóveis legítimos de alto valor; cortes/winsorização degradam drasticamente $R^2$ e RMSE. |
| **Variável `sfh`** | `remover` | Reduz MAE em ~R$ 4.953 sem perda de informação explicativa. |
| **Encoding (`F2-1`/`F2-2`)** | `target_forte_100` | Reduz dimensionalidade de 131,7 para 17 colunas, elevando $R^2$ para `0,7754` e reduzindo MAE para `R$ 172.919,41`. |