# Projeto CIN0208 — Predição de Valor de Imóvel (kNN)

## Sobre

Projeto da disciplina Ciência de Dados (CIN0208 — CIn UFPE, 2026.2). Aplicação do
algoritmo **kNN** à predição do valor de avaliação de imóveis em Recife, usando dados
do ITBI (Imposto sobre Transmissão de Bens Imóveis) de 2025.

Problema formulado como tarefa de **regressão**.

## Integrantes

- [Nome 1]
- [Nome 2]
- [Nome 3]
- [Nome 4]
- [Nome 5]

## Datas importantes

- **08/10** — Checkpoint 1 (apresentação de andamento)
- **19/11** — Acompanhamento final
- **26/11** — Apresentação final

## Dataset

- Fonte: ITBI Recife 2025
- ~16.300 instâncias, 22 atributos
- Target: `valor_avaliacao`

## Estrutura do repositório

```
projeto-itbi/
├── data/
│   ├── raw/            # dataset original, sem modificações
│   └── processed/      # dataset(s) após limpeza/preprocessing
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_encoding_selecao.ipynb
│   ├── 03_normalizacao_pca.ipynb
│   ├── 04_pipeline_experimentos.ipynb
│   └── 05_analise_resultados.ipynb
├── src/                 # funções compartilhadas (preprocessing, métricas, etc.)
├── resultados/           # tabela consolidada dos 150+ experimentos
├── requirements.txt
└── README.md
```

## Ambiente

Ver `requirements.txt`. Seed fixa usada em todo o projeto: `random_state=42`.

## Decisões do projeto

Registradas nas Issues deste repositório (uma por decisão relevante de
pré-processamento/modelagem), com o formato:

> Decidi X porque Y. Alternativa considerada: Z (descartada porque ...).

## Divisão de trabalho

| Etapa | Responsável |
|---|---|
| 1. EDA + pré-processamento base | |
| 2. Encoding + seleção de atributos | |
| 3. Normalização + redução de dimensionalidade | |
| 4. Pipeline experimental + kNN | |
| 5. Análise de resultados + relatório | |
