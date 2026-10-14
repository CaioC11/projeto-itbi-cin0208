<!--Capsule render cima-->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&height=130&color=0:FF8C00,100:FF6B35&v=99&version=3&cache=1" width="100%" />
</p>


---

<!-- Subtítulo Disciplina -->
<h1 align="center" style="border-bottom: none !important; padding-bottom: 0 !important; margin-top: 20px;">Predição de Valor de Imóvel com kNN</h1>
<h3 align="center" style="border-bottom: none !important; padding-bottom: 0 !important; color: #666;">Ciência de Dados — CIN0208 (UFPE, 2026.2)</h3>
<br>

<!--Capsule render cima área do sumário e resumo-->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&height=130&color=0:FF8C00,100:FF6B35&v=99&version=3&cache=1" width="100%" />
</p>

<h2 style="border-bottom: none !important; padding-bottom: 0 !important;"> Sumário</h2>

* [Resumo do Projeto](#resumo)
* [Guia de Reprodução](#guia-de-reprodução)
* [Arquitetura do Projeto](#arquitetura-do-projeto)
* [Dataset](#dataset-e-contexto)
* [Tecnologias Utilizadas](#tecnologias-utilizadas)
* [Equipe](#equipe-desenvolvimento)

---

<a id="resumo"></a>
<h2 style="border-bottom: none !important; padding-bottom: 0 !important;"> Resumo</h2>

> Projeto de **Ciência de Dados** desenvolvido como desafio da disciplina **CIN0208** do Centro de Informática da UFPE, focado na **previsão de valor de avaliação imobiliária** em Recife utilizando o algoritmo **k-Nearest Neighbors (kNN)** como regressor.
>
> A solução foi construída a partir de dados do **ITBI (Imposto sobre Transmissão de Bens Imóveis) de 2025**, cobrindo desde a **análise exploratória (EDA) das variáveis alvo, numéricas e categóricas**, passando pela **identificação de padrões espaciais** (localização como atributo candidato), até chegar à **análise das escalas e das opções de pré-processamento** para a futura modelagem. O pipeline de treinamento e validação cruzada ainda não foi entregue nas branches disponíveis.
>
> O projeto demonstra que **área construída tem correlação de Pearson de aproximadamente 0,92 com o valor de avaliação**. A análise espacial revela **heterogeneidade regional** (zonas como São José e Santo Antônio apresentam variações de **R$ 7,5 mil vs R$ 1,5 mil por m²**). A análise das distâncias evidencia a necessidade de normalização: sem ela, `sfh` responde por aproximadamente 95,5% da distância euclidiana quadrática média na amostra, enquanto a contribuição de `area_construida` fica próxima de zero.
>
> As análises entregues abrangem **caracterização exploratória completa**, **propostas de seleção de atributos** e **comparação das contribuições à distância antes e depois da padronização**. A amostragem da EDA usa `random_state=42`; os experimentos com múltiplos valores de k e a validação cruzada ficam para a etapa de modelagem.

---

<!--Capsule render cima área do guia de instação-->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&height=130&color=0:FF8C00,100:FF6B35&v=99&version=3&cache=1" width="100%" />
</p>

<a id="guia-de-reprodução"></a>
<h2 style="border-bottom: none !important; padding-bottom: 0 !important;"> Guia de Reprodução</h2>

> **Ambiente de Execução:** Python **3.10 a 3.12**, com análises em Jupyter Notebooks e seed 42 nas amostragens. A integração foi validada com Python 3.12.

### Pré-requisitos do Sistema
* **Python 3.10+** (recomendado 3.11 ou 3.12)
* **Git** (para clonagem do repositório)
* **pip** (gerenciador de pacotes Python)

### Instalação Passo a Passo

#### Passo 1: Clone o Repositório
```bash
git clone https://github.com/CaioC11/projeto-itbi-cin0208.git
cd projeto-itbi-cin0208
```

#### Passo 2: Crie um Ambiente Virtual (Recomendado)
```bash
# Em Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Em Windows
python -m venv venv
venv\Scripts\activate
```

#### Passo 3: Instale as Dependências
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Validação:** Após a instalação, teste se tudo foi importado corretamente:
```bash
python -c "import pandas as pd; import sklearn; import matplotlib; print('✓ Dependências instaladas com sucesso!')"
```

---

### Carregamento de Dados

O projeto utiliza um módulo centralizado para leitura padronizada dos dados, **sempre use `carregar_dados()`**:

```python
from src.carregar_dados import carregar_dados

# Carrega o CSV tratando separador (;) e decimais (vírgula/ponto)
df = carregar_dados()
print(df.shape)  # (16318, 22)
```

**Nunca use `pd.read_csv()` diretamente** — o CSV possui separador `;` e mistura vírgula/ponto decimal.

---

### Executar as análises

Abra `jupyter notebook` a partir da raiz do repositório e execute, do início ao fim, os três notebooks em `notebooks/`. Cada notebook também funciona com o diretório de trabalho em `notebooks/` e utiliza o mesmo carregador.

Para validar todos de forma automática:

```bash
python scripts/validar_notebooks.py
```

A validação salva cópias executadas em `resultados/notebooks/`, sem alterar os notebooks de autoria. As pastas `data/processed/` e `resultados/` são reservadas para artefatos; não há dataset processado ou métricas de treinamento entregues ainda.

### Validação Pós-Execução

Após completar a EDA, os seguintes artefatos devem estar presentes:

**Estrutura de Diretórios Criada:**
```
data/
├── raw/
│   └── raw_csv_itbi.csv
└── processed/
    └── .gitkeep

notebooks/
├── 01a_eda_alvo.ipynb
├── 01b_eda_numericas.ipynb
└── 01c_eda_categoricas.ipynb

resultados/
└── .gitkeep
```

**Métricas e Artefatos Esperados:**
- Distribuição da variável alvo (`valor_avaliacao`)
- Correlação de area_construida com o valor (Pearson ~0.92)
- Comparação das medianas de valor por m² por bairro (São José vs Santo Antônio: **até 5x diferença**)
- Impacto da normalização na contribuição de cada variável à distância quadrática

**Etapa pendente:** baseline kNN, múltiplos valores de k e validação cruzada com métricas de regressão.

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&height=130&color=0:FF8C00,100:FF6B35&v=99&version=3&cache=1" width="100%" />
</p>

<a id="arquitetura-do-projeto"></a>
<h2 style="border-bottom: none !important; padding-bottom: 0 !important;"> Arquitetura do Projeto</h2>

```text
projeto-itbi-cin0208/
├── data/
│   ├── raw/
│   │   └── raw_csv_itbi.csv
│   └── processed/
│       └── .gitkeep
├── notebooks/
│   ├── 01a_eda_alvo.ipynb
│   ├── 01b_eda_numericas.ipynb
│   └── 01c_eda_categoricas.ipynb
├── src/
│   ├── __init__.py
│   └── carregar_dados.py
├── resultados/
│   └── .gitkeep
├── Docs/
│   ├── decisoes.md
│   └── cronograma.md
├── scripts/
│   └── validar_notebooks.py
├── requirements.txt
├── .gitignore
└── README.md
```

**Convenções do Projeto:**
- Uma branch por tarefa: `eda-alvo`, `eda-numericas`, `eda-categoricas`, `pipeline`
- Um notebook por pessoa responsável
- Merge na `main` apenas com notebooks rodando do zero
- Seed fixa: `random_state=42` em **todas** as operações estocásticas

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&height=130&color=0:FF8C00,100:FF6B35&v=99&version=3&cache=1" width="100%" />
</p>

<a id="dataset-e-contexto"></a>
<h2 style="border-bottom: none !important; padding-bottom: 0 !important;"> Dataset</h2>

- **Fonte:** ITBI (Imposto sobre Transmissão de Bens Imóveis) — Recife, 2025
- **Instâncias:** ~16.300 transações imobiliárias
- **Atributos:** 22 variáveis (numéricas e categóricas)
- **Target:** `valor_avaliacao` (valor predito em R$)

### Variáveis-Chave Identificadas
| Variável | Tipo | Relevância | Observação |
|---|---|---|---|
| `area_construida` | Numérica | **Crítica** | Pearson 0.92 com o valor; principal preditor |
| `latitude` / `longitude` | Numérica | **Alta** | Localização regional; kNN decide por proximidade |
| `sfh` | Numérica | **Baixa** | Correlação −0,03 com preço; ~95,5% da distância euclidiana sem normalização |
| `bairro` | Categórica | **Alta** | São José e Santo Antônio: **R$ 7,5 mil vs R$ 1,5 mil por m²** |
| `area_terreno` | Numérica | **Candidata** | Spearman −0,85 com `fracao_ideal`; mantida na EDA para comparação futura |

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&height=130&color=0:FF8C00,100:FF6B35&v=99&version=3&cache=1" width="100%" />
</p>

<a id="tecnologias-utilizadas"></a>

<h1 align="center" style="border-bottom: none !important; padding-bottom: 0 !important;">Stack Tools</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff&style=for-the-badge" alt="Python Badge" />&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=fff&style=for-the-badge" alt="Pandas Badge" />&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=fff&style=for-the-badge" alt="NumPy Badge" />&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=fff&style=for-the-badge" alt="Scikit-learn Badge" />&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://img.shields.io/badge/Matplotlib-11557C?logo=python&logoColor=fff&style=for-the-badge" alt="Matplotlib Badge" />&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://img.shields.io/badge/Jupyter-F37726?logo=jupyter&logoColor=fff&style=for-the-badge" alt="Jupyter Badge" />&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://img.shields.io/badge/Git-F03C2E?logo=git&logoColor=fff&style=for-the-badge" alt="Git Badge" />
</p>

<br>

### Dependências Principais

Veja `requirements.txt` para a lista completa. Principais pacotes:

- **pandas** (2.2.2): Manipulação e análise de dados
- **numpy** (1.26.4): Computação numérica
- **scikit-learn** (1.5.1): Algoritmos de ML, validação cruzada e normalização
- **matplotlib** (3.9.0): Visualizações estáticas
- **seaborn** (0.13.2): Visualizações estatísticas
- **jupyter** (1.0.0): Ambiente interativo para análise

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&height=130&color=0:FF8C00,100:FF6B35&v=99&version=3&cache=1" width="100%" />
</p>

<a id="equipe-desenvolvimento"></a>
<h2 align="center" style="border-bottom: none !important; padding-bottom: 0 !important;">👥 Equipe de Desenvolvimento</h2>

<div align="center">
  <table border="0" cellpadding="0" cellspacing="0" style="border-collapse: collapse; border: none;">
    <tr>
      <td valign="middle" align="center" style="border: none; padding: 0 10px;">
        <a href="https://github.com/Eugeni1">
          <img src="https://images.weserv.nl/?url=github.com/Eugeni1.png&w=60&h=60&mask=circle" alt="Eugeni1"><br><strong>Eugeni1</strong>
        </a>
      </td>
      <td valign="middle" align="center" style="border: none; padding: 0 10px;">
        <a href="https://github.com/CaioC11">
          <img src="https://images.weserv.nl/?url=github.com/CaioC11.png&w=60&h=60&mask=circle" alt="CaioC11"><br><strong>CaioC11</strong>
        </a>
      </td>
      <td valign="middle" align="center" style="border: none; padding: 0 10px;">
        <a href="https://github.com/GabRamosBr">
          <img src="https://images.weserv.nl/?url=github.com/GabRamosBr.png&w=60&h=60&mask=circle" alt="GabRamosBr"><br><strong>GabRamosBr</strong>
        </a>
      </td>
      <td valign="middle" align="center" style="border: none; padding: 0 10px;">
        <a href="https://github.com/NyckolasJGF">
          <img src="https://images.weserv.nl/?url=github.com/NyckolasJGF.png&w=60&h=60&mask=circle" alt="NyckolasJGF"><br><strong>NyckolasJGF</strong>
        </a>
      </td>
      <td valign="middle" align="center" style="border: none; padding: 0 10px;">
        <a href="https://github.com/FabinhoFarias">
          <img src="https://images.weserv.nl/?url=github.com/FabinhoFarias.png&w=60&h=60&mask=circle" alt="FabinhoFarias"><br><strong>FabinhoFarias</strong>
        </a>
      </td>
    </tr>
  </table>
</div>

<br>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&height=130&color=0:FF8C00,100:FF6B35&v=99&section=footer&width=100%" alt="Footer Animado" />
</p>
