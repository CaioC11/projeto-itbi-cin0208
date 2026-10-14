# Plano de Experimentos: Projeto ITBI Recife (CIN0208)
 
Versão do checkpoint de 14/10/2026. Baseado nas EDAs `01a_eda_alvo`, `01b_eda_numericas` e `01c_eda_categoricas`. Os pontos marcados com **[a confirmar]** ainda dependem de decisão do grupo ou de resposta da professora.
 
## 1. Problema e protocolo
 
- **Tarefa:** regressão, prevendo `valor_avaliacao` (valor de avaliação do imóvel, ITBI Recife 2025).
- **Modelo:** somente kNN (`KNeighborsRegressor`) com a configuração padrão do scikit-learn (k=5, distância euclidiana, pesos uniformes) nos experimentos principais, como exige o enunciado. Nenhum outro modelo preditivo entra no projeto, nem como auxiliar de pré-processamento (a seleção de atributos usa apenas métodos estatísticos).
- **Validação:** 10-fold (`KFold`, `shuffle=True`, `random_state=42`). Não é estratificada, pois o problema é de regressão. Os mesmos folds valem para todos os experimentos, o que permite comparação pareada.
- **Sem vazamento de dados:** toda etapa que aprende com os dados (imputação, encoding, escala, seleção, PCA, limites de outliers) é ajustada só no treino de cada fold.
- **Alvo em log:** o modelo treina com `np.log1p(valor_avaliacao)` e as predições voltam a R$ com `np.expm1` (decisão da EDA do alvo: assimetria 32,83 → 0,64). Qualquer etapa que use o alvo, como o Target Encoding, enxerga o alvo em log.
- **Métricas:** MAE, MAPE, RMSE e R², com média e desvio-padrão entre os 10 folds, calculadas em R$ e, complementarmente, em log. Como o alvo é muito desbalanceado (média R$ 659 mil contra mediana R$ 350 mil), o erro também é reportado **por faixa de valor** (por exemplo, por quartil do alvo e para o 1% mais caro), pois MAE e RMSE globais escondem o desempenho nas caudas.
- **Ambiente:** versões das bibliotecas fixadas no `requirements.txt` e citadas no relatório.
## 2. Dados e pré-tratamento fixo
 
Base bruta: 16.318 instâncias e 22 colunas.
 
**Colunas descartadas:** `logradouro`, `numero`, `complemento`, `cod_logradouro`, `data_transacao`, `ano` (constante), `cidade` e `uf` (constantes).
 
**Atributos mantidos (13):**
 
| Tipo | Atributos |
|---|---|
| Numéricos (7) | `ano_construcao`, `area_terreno`, `area_construida`, `fracao_ideal`, `sfh`, `latitude`, `longitude` |
| Categóricos (6) | `bairro` (87 categorias), `tipo_imovel` (18), `tipo_construcao` (16), `padrao_acabamento` (3, ordinal), `estado_conservacao` (3), `tipo_ocupacao` (3) |
 
**Pré-tratamento determinístico, aplicado antes da validação cruzada:**
- `area_terreno == 0` e `fracao_ideal == 0` viram faltantes (a EDA indica dado ausente registrado como zero).
- Grafias de `bairro` padronizadas e espaços extras removidos (ver `01c_eda_categoricas`).
- **Remoção das 301 linhas exatamente duplicadas** (1,8%, repetindo todas as colunas). No kNN, uma cópia no treino e outra no teste cria um vizinho a distância zero e infla o desempenho.
- **Remoção do registro de R$ 19,52** (apartamento de 62,67 m² em Boa Viagem), erro de registro incompatível com o mercado, que também distorceria o MAPE.
- Depois dessas remoções a base fica com cerca de 16.000 instâncias.
## 3. Ordem das etapas no pipeline
 
Cada experimento aplica as etapas nesta ordem, todas ajustadas só no treino do fold:
 
1. Valores ausentes (remoção das linhas ou imputação)
2. Encoding das categóricas
3. Escalonamento
4. Seleção de atributos (sobre as colunas já codificadas e escaladas)
5. Redução de dimensionalidade (PCA)
6. kNN
Observação: como a seleção atua depois do encoding, o número de colunas muda com a estratégia. Com One-Hot há cerca de 130 colunas (só `bairro` responde por 87), então top-25% mantém umas 33. Com Ordinal, Frequency ou Target Encoding são 13 colunas, e top-25% mantém só 3. Isso é esperado e deve ser discutido nos resultados, porque o efeito da seleção depende do encoding.
 
## 4. Grade de experimentos principal
 
| Etapa | Variações | Nº |
|---|---|---|
| Valores ausentes (`latitude`/`longitude` 36%; `area_terreno` zerada) | (a) remoção das linhas com faltantes; (b) imputação pela mediana | 2 |
| Encoding das categóricas | One-Hot; Ordinal/Label; Frequency; Target Encoding | 4 |
| Escalonamento | sem escala; Min-Max; Z-score | 3 |
| Seleção de atributos | todos; top-50% por informação mútua; top-25% por informação mútua; por correlação; top-50% por teste F | 5 |
| Redução de dimensionalidade | sem PCA; PCA 90%; PCA 95%; PCA 99% de variância | 4 |
 
**Total: 2 × 4 × 3 × 5 × 4 = 480 configurações distintas** (mínimo exigido: 150), o que dá 4.800 ajustes do kNN (480 × 10 folds). O balanceamento de classes do enunciado não se aplica, porque o problema é de regressão.
 
### Definições das etapas
 
**Valores ausentes**
- *Remoção:* descarta as linhas com coordenadas ou `area_terreno` faltantes. Isso muda o conjunto avaliado (a perda é grande, pois as coordenadas faltam em 36%), então o relatório deve discutir a diferença, e a comparação mais justa entre remoção e imputação avalia ambas no mesmo subconjunto de linhas. **[a confirmar]**
- *Imputação pela mediana* (e não pela média, por causa da assimetria). Cuidado: imputar 36% das coordenadas com a mediana global concentra esses imóveis num ponto do mapa e cria vizinhos falsos no kNN. Se a professora indicar, uma variante é imputar pela mediana do bairro, que preserva a localização aproximada. **[a confirmar]**
**Encoding** (aplicado às categóricas nominais; `padrao_acabamento` é ordinal em todas as variantes: Simples < Médio < Superior)
- *One-Hot:* `handle_unknown="ignore"`, sem ordem artificial, mas gera muitas colunas.
- *Ordinal/Label:* cada categoria vira um inteiro. É a variante mais barata, mas impõe uma ordem artificial em categorias nominais como `bairro`, o que distorce as distâncias do kNN. Deve ser discutido nos resultados.
- *Frequency:* cada categoria é substituída pela sua frequência relativa no treino (categoria não vista no teste recebe 0). Gera uma coluna por atributo e não usa o alvo, mas categorias diferentes com a mesma frequência ficam indistinguíveis.
- *Target Encoding:* cada categoria é substituída por uma média do alvo em log, com suavização (`TargetEncoder` do scikit-learn, que usa validação cruzada interna ao ajustar o treino e assim reduz o vazamento). Embute a informação do alvo numa coluna, o que costuma ajudar o kNN com categorias de alta cardinalidade como `bairro`, mas é a variante com maior risco de vazamento se mal implementada, então só pode ser ajustada dentro do fold.
**Escalonamento**
- Sem escala, Min-Max e Z-score (`StandardScaler`). Sem escala o `sfh` (amplitude de 4.000.000) responde por 95,5% da distância do kNN, e essa combinação fica na grade de propósito para medir o efeito da escala. O PCA vem depois da escala, e a combinação PCA sem escala também é um experimento legítimo.
**Seleção de atributos** (apenas métodos estatísticos, sem usar outro modelo preditivo)
- *Todos:* referência sem seleção.
- *Top-50% e top-25% por informação mútua* (`SelectKBest` com `mutual_info_regression`, `random_state=42`): capta relações não lineares entre cada atributo e o alvo.
- *Por correlação:* calcula o |Spearman| de cada atributo com o alvo, remove entre pares com |ρ| > 0,85 o atributo menos correlacionado com o alvo (a EDA aponta `area_terreno` × `fracao_ideal` com ρ = −0,85) e descarta atributos com |ρ| < 0,05. Limiares a ajustar no notebook 02. **[a confirmar]**
- *Top-50% por teste F* (`SelectKBest` com `f_regression`): capta apenas relação linear e serve de contraponto à informação mútua.
- Todos os critérios são calculados no treino de cada fold. O RReliefF (que pesa os atributos pela diferença entre vizinhos, combinando bem com o kNN) fica como extra opcional na fase 2, se a biblioteca (`skrebate`) instalar e rodar no ambiente do grupo.
**PCA**
- Sem PCA, ou PCA retendo 90%, 95% e 99% da variância explicada, ajustado no treino do fold depois do encoding e da escala.
## 5. Fase 2: análises complementares (fora das 480 configurações)
 
Rodam só nas **10 melhores configurações** da grade principal (menor MAE médio em R$, a confirmar), para controlar o custo computacional:
 
1. **Tratamento de outliers.** A EDA decidiu manter os valores altos, e a grade principal parte disso (referência). Aqui comparamos três níveis:
   - *Manter tudo* (já medido na grade principal).
   - *Remover do treino* as linhas com alvo fora do IQR calculado na escala log (cerca de 4,6% dos imóveis; o IQR na escala original marcaria 10,16% porque a própria assimetria o infla). A remoção vale **só no treino de cada fold**, com os limites calculados no treino; o teste permanece intacto, para que as métricas continuem comparáveis.
   - *Winsorização* dos atributos numéricos nos percentis 1 e 99 do treino (limites aprendidos no treino e aplicados também ao teste).
   Hipótese a verificar: num problema de regressão desbalanceada, os valores altos são justamente os casos raros que importam, então removê-los pode piorar o erro nessa faixa mesmo melhorando o erro médio. Por isso o desempenho por faixa de valor é essencial nesta análise.
2. **Alvo sem log vs. com log**, para quantificar o ganho da transformação.
3. **Variação de k e da métrica de distância** (por exemplo, k de 3 a 25; Euclidiana e Manhattan). É análise extra, e a grade principal mantém o padrão.
4. **Reamostragem para regressão** (SMOTER/SMOGN), aplicada só no treino de cada fold, para tratar o desbalanceamento do alvo.
5. **Custo computacional:** tempo de ajuste e de predição por configuração.
Total da fase 2 (itens 1 a 3 para 10 configurações): cerca de 20 + 10 + algumas dezenas de execuções, contadas à parte dos 480 principais.
 
## 6. Implementação e custo
 
- **Grade principal:** `Pipeline` do scikit-learn dentro de `cross_validate`, com `TransformedTargetRegressor` para o log. Os IDs são gerados pelo produto cartesiano da grade (`itertools.product`).
- **Fase 2:** remoção de outliers do treino e SMOGN alteram as linhas do treino, o que um `Pipeline` comum não faz. Por isso, nesta fase, usar um laço de validação cruzada próprio (mesmos folds e mesma semente), com as funções de pré-processamento aplicadas dentro do laço.
- **Custo:** a informação mútua é a etapa mais cara. Para evitar recalcular, usar o parâmetro `memory` do `Pipeline` (cache) ou calcular as pontuações uma vez por fold e por combinação de ausentes e encoding, reaproveitando entre escala e PCA. Antes de rodar tudo, medir o tempo de uma configuração e multiplicar por 480.
## 7. Pontos em aberto
 
Levar ao checkpoint como dúvidas para a professora:
 
- **Faltantes de coordenada (36%):** remover essas linhas ou imputar (mediana global ou por bairro)? A remoção muda o conjunto avaliado, então a escolha afeta a comparação entre as estratégias.
- **Alvo em log:** qual o impacto da transformação no kNN e como justificá-la no relatório? A EDA mostra assimetria de 32,83 → 0,64, e o log será comparado com o alvo original na fase 2.
- **Enquadramento como regressão desbalanceada:** o alvo tem cauda longa à direita (média R$ 659 mil contra mediana R$ 350 mil, 1% dos imóveis acima de R$ 4,83 milhões). Apresentar o problema como regressão desbalanceada, com SMOTER/SMOGN e tratamento de outliers como técnicas a comparar, e perguntar se essa abordagem faz sentido para a disciplina.
Decisões internas do grupo:
 
- **`sfh`:** 65% de zeros e correlação próxima de zero. A EDA recomenda testar com e sem ele, o que a seleção de atributos já cobre.
- **Imóveis do mesmo prédio:** unidades do mesmo edifício têm coordenadas e áreas quase idênticas e podem cair em folds diferentes. Registrar como limitação no relatório; um `GroupKFold` por prédio seria um refinamento futuro.
## 8. Organização no repositório
 
| Notebook / arquivo | Conteúdo |
|---|---|
| `02_encoding_selecao` | Tratamento de faltantes, os 4 encodings e os métodos de seleção |
| `03_normalizacao_pca` | Escalonamento e PCA |
| `04_pipeline_experimentos` | Montagem do `Pipeline` e execução das 480 configurações |
| `05_analise_resultados` | Tabelas, gráficos, desempenho por faixa de valor e fase 2 (outliers, log, k, SMOGN) |
| `resultados/experimentos.csv` | Uma linha por configuração, com colunas `id`, `ausentes`, `encoding`, `escala`, `selecao`, `pca`, e `media` e `desvio` de cada métrica |
 
Cada configuração recebe um ID (`exp_001` a `exp_480`), o que permite comprovar no relatório que o total é ≥ 150.
 
## 9. Cronograma (proposta)
 
| Período | Entrega |
|---|---|
| 14/10 | Checkpoint 1: EDA e plano de experimentos |
| 15/10 a 25/10 | Pipeline completo, baseline (média) e kNN k=5 (notebooks 02 a 04); medição do tempo por configuração |
| 26/10 a 08/11 | Execução das 480 configurações e coleta dos resultados |
| 09/11 a 18/11 | Fase 2 (outliers, log, k, SMOGN), análise, gráficos e rascunho do relatório |
| 19/11 | Acompanhamento final: resultados e discussão preliminares |
| 20/11 a 25/11 | Ajustes, revisão do relatório e slides |
| 26/11 | Apresentação final e entrega do relatório |