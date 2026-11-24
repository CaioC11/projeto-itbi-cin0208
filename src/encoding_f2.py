import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FrequencyEncoding(BaseEstimator, TransformerMixin):
    """Substitui cada categoria pela sua frequÃªncia relativa no treino."""

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
                serie = serie.where(serie.isin(categorias), 'outras')
                categorias = categorias + ['outras']

            for categoria in categorias:
                nome = f'{col}_{categoria}'
                novas_colunas[nome] = (serie == categoria).astype(float) * self.peso

        dummies = pd.DataFrame(novas_colunas, index=X.index)
        return pd.concat([X.drop(columns=self.colunas), dummies], axis=1)

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
                serie = serie.where(serie.isin(categorias), 'outras')
                categorias = categorias + ['outras']

            for categoria in categorias:
                nome = f'{col}_{categoria}'
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

class TargetEncodingSuave(BaseEstimator, TransformerMixin):
    """Target Encoding com suavizaÃ§Ã£o em direÃ§Ã£o Ã  mÃ©dia global do treino."""

    def __init__(self, colunas, suavizacao=10):
        self.colunas = colunas
        self.suavizacao = suavizacao

    def fit(self, X, y):
        y = pd.Series(np.asarray(y), index=X.index)

        self.media_global_ = y.mean()
        self.mapas_ = {}

        for col in self.colunas:
            grupos = y.groupby(X[col]).agg(['mean', 'count'])

            mapa = (
                grupos['count'] * grupos['mean']
                + self.suavizacao * self.media_global_
            ) / (grupos['count'] + self.suavizacao)

            self.mapas_[col] = mapa

        return self

    def transform(self, X):
        X = X.copy()

        for col in self.colunas:
            X[col] = X[col].map(self.mapas_[col]).fillna(self.media_global_)

        return X

class TratamentoFaltantes(BaseEstimator, TransformerMixin):

    def __init__(self, estrategia='mediana_bairro'):
        estrategias_validas = (
            'remover_linhas',
            'mediana_bairro',
            'mediana_bairro_indicador',
            'sem_lat_lon',
            'area_terreno_tipo',
            'media'
        )
        if estrategia not in estrategias_validas:
            raise ValueError(f'EstratÃ©gia invÃ¡lida: {estrategia}')
        self.estrategia = estrategia

    def fit(self, X, y=None):
        self.cols_numericas_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.cols_com_nulos_ = [c for c in self.cols_numericas_ if X[c].isna().any()]
        self.mediana_global_ = X[self.cols_numericas_].median()
        self.media_global_ = X[self.cols_numericas_].mean()

        if 'bairro' in X.columns:
            self.mediana_por_bairro_ = X.groupby('bairro')[self.cols_numericas_].median()
        else:
            self.mediana_por_bairro_ = None

        if 'tipo_imovel' in X.columns and 'area_terreno' in X.columns:
            self.mediana_terreno_por_tipo_ = X.groupby('tipo_imovel')['area_terreno'].median()
        else:
            self.mediana_terreno_por_tipo_ = None

        self.cols_lat_lon_ = [
            c for c in X.columns if 'lat' in c.lower() or 'lon' in c.lower()
        ]
        return self

    def transform(self, X):
        X = X.copy()

        if self.estrategia == 'remover_linhas':
            return X.dropna()

        if self.estrategia == 'media':
            for col in self.cols_numericas_:
                if col in X.columns:
                    X[col] = X[col].fillna(self.media_global_[col])
            return X

        if self.estrategia == 'sem_lat_lon':
            X = X.drop(columns=[c for c in self.cols_lat_lon_ if c in X.columns])
            for col in self.cols_numericas_:
                if col in X.columns:
                    X[col] = X[col].fillna(self.mediana_global_[col])
            return X

        if self.estrategia == 'area_terreno_tipo':
            if 'area_terreno' in X.columns and self.mediana_terreno_por_tipo_ is not None and 'tipo_imovel' in X.columns:
                mapa_terreno = X['tipo_imovel'].map(self.mediana_terreno_por_tipo_)
                X['area_terreno'] = X['area_terreno'].fillna(mapa_terreno).fillna(self.mediana_global_['area_terreno'])
            for col in self.cols_numericas_:
                if col in X.columns and col != 'area_terreno':
                    X[col] = X[col].fillna(self.mediana_global_[col])
            return X

        if self.estrategia in ('mediana_bairro', 'mediana_bairro_indicador'):
            if self.estrategia == 'mediana_bairro_indicador':
                for col in self.cols_com_nulos_:
                    if col in X.columns:
                        X[f'{col}_missing'] = X[col].isna().astype(int)

            for col in self.cols_numericas_:
                if col not in X.columns:
                    continue
                if self.mediana_por_bairro_ is not None and 'bairro' in X.columns:
                    mapa_bairro = X['bairro'].map(self.mediana_por_bairro_[col])
                    X[col] = X[col].fillna(mapa_bairro)
                X[col] = X[col].fillna(self.mediana_global_[col])
            return X


class TargetEncodingForte(TargetEncodingSuave):
    """Target Encoding com suavização mais forte (padrão: 100)."""

    def __init__(self, colunas, suavizacao=100):
        super().__init__(colunas, suavizacao)

