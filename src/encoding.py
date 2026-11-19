"""Transformadores de encoding e montagem de pipeline (F2 - Projeto ITBI, CIN0208)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ORDEM_ACABAMENTO = {"padrao_acabamento": ["Simples", "Médio", "Superior"]}
ORDENS_ORDINAL = {
    "padrao_acabamento": ["Simples", "Médio", "Superior"],
    "estado_conservacao": ["Mau", "Regular", "Bom"],
}
COLUNAS_ALTA_CARDINALIDADE = ["bairro", "tipo_imovel", "tipo_construcao"]
COLUNAS_BAIXA_CARDINALIDADE = ["tipo_ocupacao", "estado_conservacao"]
COLUNAS_NOMINAIS_ORDINAL = ["bairro", "tipo_imovel", "tipo_construcao", "tipo_ocupacao"]


class OneHotRaras(BaseEstimator, TransformerMixin):
    """One-Hot Encoding com categorias aprendidas somente no fit."""

    def __init__(self, colunas, min_count=None):
        self.colunas = colunas
        self.min_count = min_count

    def fit(self, X, y=None):
        self.categorias_ = {}
        self.agrupa_ = {}

        for col in self.colunas:
            contagem = X[col].value_counts(dropna=False)
            if self.min_count is None:
                manter = contagem
            else:
                manter = contagem[contagem >= self.min_count]

            self.categorias_[col] = list(manter.index)
            self.agrupa_[col] = len(manter) < len(contagem)

        return self

    def transform(self, X):
        X = X.copy()
        novas_colunas = {}

        for col in self.colunas:
            serie = X[col]
            categorias = self.categorias_[col]

            if self.agrupa_[col]:
                serie = serie.where(serie.isin(categorias), "outras")
                categorias = categorias + ["outras"]

            for categoria in categorias:
                nome = f"{col}_{categoria}"
                novas_colunas[nome] = (serie == categoria).astype(int)

        dummies = pd.DataFrame(novas_colunas, index=X.index)
        return pd.concat([X.drop(columns=self.colunas), dummies], axis=1)


class OrdinalOrdenado(BaseEstimator, TransformerMixin):
    """Codifica categorias segundo uma ordem definida manualmente."""

    def __init__(self, ordens):
        self.ordens = ordens

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col, ordem in self.ordens.items():
            mapa = {categoria: i + 1 for i, categoria in enumerate(ordem)}
            X[col] = X[col].map(mapa).fillna(0).astype(int)
        return X


class FrequencyEncoding(BaseEstimator, TransformerMixin):
    """Substitui cada categoria pela sua frequência relativa no treino."""

    def __init__(self, colunas):
        self.colunas = colunas

    def fit(self, X, y=None):
        self.freq_ = {
            col: X[col].value_counts(normalize=True)
            for col in self.colunas
        }
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.colunas:
            X[col] = X[col].map(self.freq_[col]).fillna(0.0)
        return X


class TargetEncodingSuave(BaseEstimator, TransformerMixin):
    """Target Encoding com suavização em direção à média global do treino."""

    def __init__(self, colunas, suavizacao=10):
        self.colunas = colunas
        self.suavizacao = suavizacao

    def fit(self, X, y):
        y = pd.Series(np.asarray(y), index=X.index)
        self.media_global_ = y.mean()
        self.mapas_ = {}

        for col in self.colunas:
            grupos = y.groupby(X[col]).agg(["mean", "count"])
            mapa = (
                grupos["count"] * grupos["mean"]
                + self.suavizacao * self.media_global_
            ) / (grupos["count"] + self.suavizacao)
            self.mapas_[col] = mapa

        return self

    def transform(self, X):
        X = X.copy()
        for col in self.colunas:
            X[col] = X[col].map(self.mapas_[col]).fillna(self.media_global_)
        return X


class TargetEncodingForte(TargetEncodingSuave):
    """Target Encoding com suavização mais forte (padrão: 100)."""

    def __init__(self, colunas, suavizacao=100):
        super().__init__(colunas, suavizacao)


class OneHotPesoReduzido(BaseEstimator, TransformerMixin):
    """One-Hot Encoding onde os valores ativos recebem um peso reduzido (< 1.0)."""

    def __init__(self, colunas, peso=0.5, min_count=None):
        self.colunas = colunas
        self.peso = peso
        self.min_count = min_count

    def fit(self, X, y=None):
        self.categorias_ = {}
        self.agrupa_ = {}

        for col in self.colunas:
            contagem = X[col].value_counts(dropna=False)
            if self.min_count is None:
                manter = contagem
            else:
                manter = contagem[contagem >= self.min_count]

            self.categorias_[col] = list(manter.index)
            self.agrupa_[col] = len(manter) < len(contagem)

        return self

    def transform(self, X):
        X = X.copy()
        novas_colunas = {}

        for col in self.colunas:
            serie = X[col]
            categorias = self.categorias_[col]

            if self.agrupa_[col]:
                serie = serie.where(serie.isin(categorias), "outras")
                categorias = categorias + ["outras"]

            for categoria in categorias:
                nome = f"{col}_{categoria}"
                novas_colunas[nome] = (serie == categoria).astype(float) * self.peso

        dummies = pd.DataFrame(novas_colunas, index=X.index)
        return pd.concat([X.drop(columns=self.colunas), dummies], axis=1)


class DropColunas(BaseEstimator, TransformerMixin):
    """Remove colunas específicas se existirem no DataFrame."""

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in self.cols if c in X.columns])


class ImputarEEscalarFinal(BaseEstimator, TransformerMixin):
    """Imputa numéricas restantes e aplica StandardScaler no final do pipeline."""

    def __init__(self, estrategia_num="median", escalar=True):
        self.estrategia_num = estrategia_num
        self.escalar = escalar

    def fit(self, X, y=None):
        self.colunas_ = list(X.columns)
        self.imputer_ = SimpleImputer(strategy=self.estrategia_num).fit(X)
        X_imp = self.imputer_.transform(X)
        self.scaler_ = StandardScaler().fit(X_imp) if self.escalar else None
        return self

    def transform(self, X):
        X_imp = self.imputer_.transform(X[self.colunas_])
        return self.scaler_.transform(X_imp) if self.scaler_ is not None else X_imp


def montar_pipeline(passos_customizados, estrategia_num="median"):
    """Monta um Pipeline sklearn compatível com src/executor.py."""
    passos = list(passos_customizados) + [("final", ImputarEEscalarFinal(estrategia_num=estrategia_num))]
    return Pipeline(passos)