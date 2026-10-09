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

- **15/10** — Checkpoint 1 (apresentação de andamento)
- **19/11** — Acompanhamento final
- **26/11** — Apresentação final

## Checkpoint 15/10: funções da semana

Trabalho assíncrono, sem reuniões, exceto o ensaio na quarta 14/10. Prazo interno: terça 13/10 à noite, com tudo no repositório.

| Função | Responsável | Entregável |
|---|---|---|
| Coordenação, repositório e apresentação | [Nome] | Repo organizado, Kanban, plano de experimentos, slides integrados |
| EDA da variável alvo (`valor_avaliacao`) | [Nome] | `01a_eda_alvo.ipynb` |
| EDA das variáveis numéricas | [Nome] | `01b_eda_numericas.ipynb` |
| EDA das categóricas e valores faltantes | [Nome] | `01c_eda_categoricas.ipynb` |
| Pipeline, baseline e primeiro kNN | [Nome] | `04_pipeline_experimentos.ipynb` |

Convenções: uma branch por tarefa (`eda-alvo`, `eda-numericas`, `eda-categoricas`, `pipeline`), um notebook por pessoa, e merge na `main` só com o notebook rodando do zero.

## Dataset

- Fonte: ITBI Recife 2025
- ~16.300 instâncias, 22 atributos
- Target: `valor_avaliacao`

## Estrutura do repositório

```
projeto-itbi-cin0208/
├── data/
│   ├── raw/            # dataset original, sem modificações
│   └── processed/      # dataset(s) após limpeza/preprocessing
├── notebooks/
│   ├── 01a_eda_alvo.ipynb
│   ├── 01b_eda_numericas.ipynb
│   ├── 01c_eda_categoricas.ipynb
│   └── 04_pipeline_experimentos.ipynb
├── src/
│   └── carregar_dados.py   # leitura padronizada do CSV (usar sempre)
├── resultados/         # tabelas e gráficos de resultados
├── requirements.txt
└── README.md
```

Os demais notebooks serão definidos após a EDA.

## Ambiente

- Python **3.10 a 3.12**.
- Instalação: `pip install -r requirements.txt`.
- Seed fixa usada em todo o projeto: `random_state=42`.
- O CSV usa `;` como separador e mistura vírgula e ponto decimal. **Carregar os dados sempre com `carregar_dados()`** de `src/carregar_dados.py`, em vez de `pd.read_csv` direto:

```python
from src.carregar_dados import carregar_dados

df = carregar_dados()
```

## Decisões do projeto

Registradas nas Issues deste repositório (uma por decisão relevante de
pré-processamento/modelagem), com o formato:

> Decidi X porque Y. Alternativa considerada: Z (descartada porque ...).

## Etapas finais (a definir após a EDA)

A divisão das etapas finais (encoding, seleção de atributos, normalização, experimentos e análise de resultados) será definida depois da EDA, possivelmente após o checkpoint de 15/10.