"""Pré-tratamento fixo e padronização de texto do dataset ITBI (Projeto ITBI, CIN0208)."""
import unicodedata

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

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


class TratamentoSFH(BaseEstimator, TransformerMixin):
    """Aplica o tratamento na coluna 'sfh': 'remover' ou 'binarizar' (sfh > 0)."""

    def __init__(self, modo="binarizar", coluna="sfh"):
        if modo not in ("remover", "binarizar"):
            raise ValueError("O modo deve ser 'remover' ou 'binarizar'.")
        self.modo = modo
        self.coluna = coluna

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.coluna not in X.columns:
            raise ValueError(f"Coluna '{self.coluna}' não encontrada no DataFrame.")

        if self.modo == "remover":
            return X.drop(columns=[self.coluna])
        elif self.modo == "binarizar":
            X[self.coluna] = (X[self.coluna].fillna(0) > 0).astype(int)
            return X


class TratamentoFaltantes(BaseEstimator, TransformerMixin):
    """Imputação ou remoção de valores faltantes ajustada exclusivamente no treino."""

    def __init__(self, estrategia="mediana_bairro"):
        estrategias_validas = (
            "remover_linhas",
            "mediana_bairro",
            "mediana_bairro_indicador",
            "sem_lat_lon",
            "area_terreno_tipo",
            "media",
        )
        if estrategia not in estrategias_validas:
            raise ValueError(f"Estratégia inválida: {estrategia}")
        self.estrategia = estrategia

    def fit(self, X, y=None):
        self.cols_numericas_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.cols_com_nulos_ = [c for c in self.cols_numericas_ if X[c].isna().any()]
        self.mediana_global_ = X[self.cols_numericas_].median()
        self.media_global_ = X[self.cols_numericas_].mean()

        if "bairro" in X.columns:
            self.mediana_por_bairro_ = X.groupby("bairro")[self.cols_numericas_].median()
        else:
            self.mediana_por_bairro_ = None

        if "tipo_imovel" in X.columns and "area_terreno" in X.columns:
            self.mediana_terreno_por_tipo_ = X.groupby("tipo_imovel")["area_terreno"].median()
        else:
            self.mediana_terreno_por_tipo_ = None

        self.cols_lat_lon_ = [
            c for c in X.columns if "lat" in c.lower() or "lon" in c.lower()
        ]
        return self

    def transform(self, X):
        X = X.copy()

        if self.estrategia == "remover_linhas":
            return X.dropna()

        if self.estrategia == "media":
            for col in self.cols_numericas_:
                if col in X.columns:
                    X[col] = X[col].fillna(self.media_global_[col])
            return X

        if self.estrategia == "sem_lat_lon":
            X = X.drop(columns=[c for c in self.cols_lat_lon_ if c in X.columns])
            for col in self.cols_numericas_:
                if col in X.columns:
                    X[col] = X[col].fillna(self.mediana_global_[col])
            return X

        if self.estrategia == "area_terreno_tipo":
            if "area_terreno" in X.columns and self.mediana_terreno_por_tipo_ is not None and "tipo_imovel" in X.columns:
                mapa_terreno = X["tipo_imovel"].map(self.mediana_terreno_por_tipo_)
                X["area_terreno"] = X["area_terreno"].fillna(mapa_terreno).fillna(self.mediana_global_["area_terreno"])
            for col in self.cols_numericas_:
                if col in X.columns and col != "area_terreno":
                    X[col] = X[col].fillna(self.mediana_global_[col])
            return X

        if self.estrategia in ("mediana_bairro", "mediana_bairro_indicador"):
            if self.estrategia == "mediana_bairro_indicador":
                for col in self.cols_com_nulos_:
                    if col in X.columns:
                        X[f"{col}_missing"] = X[col].isna().astype(int)

            for col in self.cols_numericas_:
                if col not in X.columns:
                    continue
                if self.mediana_por_bairro_ is not None and "bairro" in X.columns:
                    mapa_bairro = X["bairro"].map(self.mediana_por_bairro_[col])
                    X[col] = X[col].fillna(mapa_bairro)
                X[col] = X[col].fillna(self.mediana_global_[col])
            return X