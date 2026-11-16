# Encoding das categóricas: experimentos

Função 2 da fase final. Código em `src/encoding.py`.

## Colunas

| Grupo | Colunas | Observação |
|---|---|---|
| Alta cardinalidade | `bairro` (87), `tipo_imovel` (18), `tipo_construcao` (16) | Muitas categorias raras (33, 10 e 6 com menos de 30 linhas) |
| Com ordem natural | `padrao_acabamento` (Simples < Médio < Superior), `estado_conservacao` (Mau < Regular < Bom) | `estado_conservacao` é 98,8% "Bom" |
| Sem ordem, baixa cardinalidade | `tipo_ocupacao` (3) | |

## O que fica fixo

- Mesmos folds de validação e métricas dos demais módulos (`random_state=42`, 10-fold)
- Alvo em log (`log1p`), com `expm1` na volta, como o pipeline definir
- kNN com a configuração padrão
- Todo encoder aprende **só com o treino de cada fold** (`fit`) e aplica no resto (`transform`)
- Categoria vista só na validação: One-Hot gera linha de zeros (ou "outras", se houver agrupamento), Frequency recebe 0, Target recebe a média global, Label recebe -1

## O que varia (um fator por vez em relação ao baseline B0)

| Exp. | O que muda em relação ao B0 | Colunas geradas pelas 6 categóricas | Pergunta |
|---|---|---|---|
| **B0** | Baseline: One-Hot nas 6 | 130 | Referência |
| **E1** | Ordinal real em `padrao_acabamento` e `estado_conservacao` | 126 | Respeitar a ordem natural ajuda? |
| **E2** | One-Hot agrupando categorias raras em "outras" (limiar: 10, 30, 100) | 84 com limiar 30 (varia por fold) | Agrupar raras reduz ruído e custo? |
| **E3** | Frequency nas 3 de alta cardinalidade | 12 | Só a frequência já basta? |
| **E4** | Target Encoding suavizado nas 3 (suavização: 1, 10, 50, 200) | 12 | O preço médio da categoria ajuda o kNN? |
| **E5** | Label arbitrário (ordem alfabética) nas 3 | 12 | A ordem inventada atrapalha? (controle negativo) |

Em E3, E4 e E5 as 3 de baixa cardinalidade seguem em One-Hot. Total de configurações: 4 sem parâmetro (B0, E1, E3, E5) + 3 de E2 + 4 de E4 = **11**.

## Como medir o efeito

Para cada experimento, registrar a média e o desvio nos 10 folds de:

- MAE, RMSE e R² (e MAPE, se o grupo adotar)
- Número de colunas após o encoding
- Tempo de `fit` + `transform`

O efeito de cada técnica é a **diferença para o B0**. O número de colunas e o tempo alimentam a discussão de custo computacional.

## Hipóteses (a confirmar com os resultados)

- E4 deve superar o B0, porque o preço médio do bairro é informativo e gera poucas colunas
- E5 deve ficar pior que E3 e E4, porque inventa uma proximidade que não existe
- E2 deve ficar perto do B0 com menos colunas

## Decisões

Registrar como Issues, no formato "Decidi X porque Y. Alternativa considerada: Z".

- Ordem do acabamento e da conservação definida à mão (Simples < Médio < Superior; Mau < Regular < Bom)
- Target Encoding suavizado, com a média da categoria puxada para a média global quanto menos imóveis ela tem
- Label arbitrário mantido só como controle negativo (E5)

## Resultados

| Exp. | Colunas | Tempo (s) | MAE | RMSE | R² |
|---|---|---|---|---|---|
| B0 | | | | | |
| E1 | | | | | |
| E2 (10 / 30 / 100) | | | | | |
| E3 | | | | | |
| E4 (1 / 10 / 50 / 200) | | | | | |
| E5 | | | | | |

## Pontos para outras funções

- **Limpeza:** 3 bairros têm grafia duplicada (87 viram 84 ao tirar acento, caixa e espaço extra), por exemplo `Cohab   Ibura`, com espaços a mais. `tipo_imovel` e `tipo_construcao` estão consistentes
- **Escala e seleção:** depois do Ordinal, do Frequency e do Target, as colunas viram números e precisam entrar na padronização, senão dominam a distância do kNN. `tipo_imovel` e `tipo_construcao` carregam quase a mesma informação (`tipo_construcao` é `tipo_imovel` mais a faixa de pavimentos), então vale decidir se uma das duas sai
- **Pipeline:** o `montar_encoder(experimento)` devolve um `Pipeline` do scikit-learn. Os encoders com alvo precisam do `y` no `fit`
