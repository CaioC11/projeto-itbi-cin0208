"""F1-3: tratamento de outliers (IQR no log via filtro_treino; winsorizacao como transformador)."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FiltroIQRLog:
    """filtro_treino(X, y): remove do treino as linhas com log1p(y) fora de [Q1 - k*IQR, Q3 + k*IQR]."""

    def __init__(self, k=1.5):
        self.k = k

    def __call__(self, X, y):
        y_arr = np.asarray(y, dtype=float).ravel()
        y_log = np.log1p(y_arr)
        q1, q3 = np.percentile(y_log, [25, 75])
        iqr = q3 - q1
        mantem = (y_log >= q1 - self.k * iqr) & (y_log <= q3 + self.k * iqr)
        posicoes = np.flatnonzero(mantem)
        return X.iloc[posicoes], y_arr[posicoes]


class Winsorizar(BaseEstimator, TransformerMixin):
    """Limita as colunas numericas aos percentis p_inf e p_sup aprendidos no treino."""

    def __init__(self, p_inf=1, p_sup=99, colunas=None):
        self.p_inf = p_inf
        self.p_sup = p_sup
        self.colunas = colunas

    def fit(self, X, y=None):
        cols = self.colunas if self.colunas is not None else list(X.select_dtypes("number").columns)
        self.colunas_ = cols
        self.limites_ = {}
        for c in cols:
            valores = X[c].to_numpy(dtype=float)
            if np.isfinite(valores).any():
                self.limites_[c] = (
                    np.nanpercentile(valores, self.p_inf),
                    np.nanpercentile(valores, self.p_sup),
                )
        return self

    def transform(self, X):
        X = X.copy()
        for c, (lo, hi) in self.limites_.items():
            X[c] = X[c].clip(lo, hi)
        return X
