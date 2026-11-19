"""Pré-tratamento fixo e padronização de texto do dataset ITBI (Projeto ITBI, CIN0208)."""
import unicodedata

import numpy as np

from src.carregar_dados import carregar_dados

ALVO = "valor_avaliacao"


def _remover_acentos(texto):
    """Remove diacríticos (acentos/cedilha) mantendo apenas caracteres ASCII."""
    normalizado = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in normalizado if not unicodedata.combining(c))


def padronizar_bairros(serie):
    """Padroniza grafias de bairro: remove espaços extras, unifica acentuação e aplica Title Case."""
    s = serie.astype("string").str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)
    s = s.map(lambda x: _remover_acentos(x) if isinstance(x, str) else x)
    s = s.str.title()
    return s.astype(object)


def limpar_dados(df):
    """Aplica o pré-tratamento fixo do plano de experimentos sobre o DataFrame bruto."""
    df = df.drop_duplicates()  # 301 linhas exatamente duplicadas
    df = df[df[ALVO] > 100].copy()  # Remove o registro de R$ 19,52
    df.loc[df["area_terreno"] == 0, "area_terreno"] = np.nan
    df.loc[df["fracao_ideal"] == 0, "fracao_ideal"] = np.nan
    if "bairro" in df.columns:
        df["bairro"] = padronizar_bairros(df["bairro"])
    return df


def carregar_e_limpar(caminho=None):
    """Lê o CSV com o carregador do repositório e aplica o pré-tratamento fixo."""
    df = carregar_dados(caminho=caminho)
    return limpar_dados(df)
