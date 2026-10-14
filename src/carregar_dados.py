"""Leitura única do CSV original do ITBI, sem descartar registros."""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
COLUNAS_NUMERICAS = (
    "numero", "valor_avaliacao", "ano_construcao", "area_terreno",
    "area_construida", "fracao_ideal", "sfh", "cod_logradouro",
    "latitude", "longitude", "ano",
)


def carregar_dados(caminho=None):
    """Carrega o dataset; caminhos relativos são resolvidos a partir da raiz.

    Campos com vírgula usam notação brasileira; campos com ponto apenas
    usam ponto decimal, como as coordenadas. Conversões inválidas geram erro.
    """
    arquivo = Path(caminho) if caminho is not None else Path("data/raw/raw_csv_itbi.csv")
    if not arquivo.is_absolute():
        arquivo = RAIZ / arquivo
    df = pd.read_csv(arquivo, sep=";", encoding="utf-8", dtype=str)
    for coluna in COLUNAS_NUMERICAS:
        texto = df[coluna].str.strip()
        brasileiro = texto.str.contains(",", regex=False, na=False)
        texto.loc[brasileiro] = (
            texto.loc[brasileiro].str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        df[coluna] = pd.to_numeric(texto, errors="raise")
    return df
