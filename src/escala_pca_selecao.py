from pathlib import Path
from functools import partial
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectKBest,
    SelectPercentile,
    VarianceThreshold,
    f_regression,
    mutual_info_regression,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline import CATEGORICAS, NUMERICAS

# colunas com assimetria positiva forte — recebem log1p antes do Z-score
COLUNAS_ASSIMETRICAS = ["area_terreno", "area_construida", "fracao_ideal", "sfh"]
COLUNAS_SIMETRICAS = [c for c in NUMERICAS if c not in COLUNAS_ASSIMETRICAS]


def _spearman_abs_seguro(x, y) -> float:
    corr = spearmanr(x, y, nan_policy="omit").correlation
    return float(np.nan_to_num(abs(corr), nan=0.0, posinf=0.0, neginf=0.0))


class Log1pSeletivoScaler(BaseEstimator, TransformerMixin):
    def __init__(self, n_assimetricas: int):
        self.n_assimetricas = n_assimetricas

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n_assim = min(self.n_assimetricas, X.shape[1])
        if n_assim > 0:
            self.scaler_assim_ = StandardScaler().fit(np.log1p(np.clip(X[:, :n_assim], a_min=0.0, a_max=None)))
        else:
            self.scaler_assim_ = None
        if X.shape[1] > n_assim:
            self.scaler_sim_ = StandardScaler().fit(X[:, n_assim:])
        else:
            self.scaler_sim_ = None
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        n_assim = min(self.n_assimetricas, X.shape[1])
        partes = []
        if self.scaler_assim_ is not None and n_assim > 0:
            partes.append(self.scaler_assim_.transform(np.log1p(np.clip(X[:, :n_assim], a_min=0.0, a_max=None))))
        if self.scaler_sim_ is not None and X.shape[1] > n_assim:
            partes.append(self.scaler_sim_.transform(X[:, n_assim:]))
        if not partes:
            return X
        return np.hstack(partes)


class SpearmanSelector(BaseEstimator, TransformerMixin):
    def __init__(self, limiar_par: float = 0.85, limiar_alvo: float = 0.05):
        self.limiar_par = limiar_par
        self.limiar_alvo = limiar_alvo

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n = X.shape[1]

        cor_alvo = np.array([_spearman_abs_seguro(X[:, j], y) for j in range(n)])
        mascara = cor_alvo >= self.limiar_alvo

        indices = list(np.where(mascara)[0])
        removidos: set[int] = set()
        for i, a in enumerate(indices):
            if a in removidos:
                continue
            for b in indices[i + 1 :]:
                if b in removidos:
                    continue
                r, _ = spearmanr(X[:, a], X[:, b], nan_policy="omit")
                r_val = float(np.nan_to_num(abs(r), nan=0.0, posinf=0.0, neginf=0.0))
                if r_val > self.limiar_par:
                    if cor_alvo[a] >= cor_alvo[b]:
                        removidos.add(b)
                    else:
                        removidos.add(a)
                        break

        self.colunas_selecionadas_ = np.array(
            [j for j in range(n) if mascara[j] and j not in removidos]
        )
        self.cor_alvo_ = cor_alvo
        return self

    def transform(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if len(self.colunas_selecionadas_) == 0:
            melhor = int(np.argmax(self.cor_alvo_))
            return X[:, [melhor]]
        return X[:, self.colunas_selecionadas_]


class PonderacaoSpearman(BaseEstimator, TransformerMixin):
    # multiplica cada coluna pelo |Spearman| com o alvo — calculado so no treino

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n = X.shape[1]
        self.pesos_ = np.array([_spearman_abs_seguro(X[:, j], y) for j in range(n)])
        return self

    def transform(self, X, y=None):
        return np.asarray(X, dtype=float) * self.pesos_


class PonderacaoMI(BaseEstimator, TransformerMixin):
    # multiplica cada coluna pela info mutua normalizada com o alvo — calculada so no treino

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        mi = mutual_info_regression(X, y, random_state=self.random_state)
        maximo = mi.max()
        self.pesos_ = mi / maximo if maximo > 0 else np.ones(len(mi))
        return self

    def transform(self, X, y=None):
        return np.asarray(X, dtype=float) * self.pesos_


def _bloco_numerico_simples(passos_escala: list) -> Pipeline:
    return Pipeline([
        ("imputa", SimpleImputer(strategy="median")),
        *passos_escala,
    ])


def _bloco_numerico_log1p_zscore() -> Pipeline:
    n_assim = len(COLUNAS_ASSIMETRICAS)
    return Pipeline([
        ("imputa", SimpleImputer(strategy="median")),
        ("log1p_zscore", Log1pSeletivoScaler(n_assimetricas=n_assim)),
    ])


def _categoricas() -> tuple:
    return ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS)


def _prep_escala(passos_num: list, passos_pos: list | None = None) -> Pipeline:
    ct = ColumnTransformer([
        ("num", _bloco_numerico_simples(passos_num), NUMERICAS),
        _categoricas(),
    ])
    etapas: list = [("ct", ct)]
    if passos_pos:
        etapas.extend(passos_pos)
    return Pipeline(etapas)


def _prep_log1p_zscore(passos_pos: list | None = None) -> Pipeline:
    colunas_ordenadas = COLUNAS_ASSIMETRICAS + COLUNAS_SIMETRICAS
    ct = ColumnTransformer([
        ("num", _bloco_numerico_log1p_zscore(), colunas_ordenadas),
        _categoricas(),
    ])
    etapas: list = [("ct", ct)]
    if passos_pos:
        etapas.extend(passos_pos)
    return Pipeline(etapas)


def experimentos_escala() -> list[dict]:
    variantes = [
        {"nome_escala": "sem_escala", "passos_num": []},
        {"nome_escala": "minmax", "passos_num": [("escala", MinMaxScaler())]},
        {"nome_escala": "robust", "passos_num": [("escala", RobustScaler())]},
        {
            "nome_escala": "yeo_johnson",
            "passos_num": [
                ("pre_escala", StandardScaler()),
                ("escala", PowerTransformer(method="yeo-johnson")),
            ],
        },
    ]

    experimentos = []
    for v in variantes:
        experimentos.append({
            "configuracao": {
                "faltantes": "mediana_global", "outliers": "nenhum",
                "sfh": "original", "encoding": "onehot",
                "escala": v["nome_escala"], "selecao": "nenhuma",
                "pca": "nenhum", "ponderacao": "nenhuma",
                "desbalanceamento": "nenhum",
            },
            "preprocessamento": _prep_escala(v["passos_num"]),
        })

    experimentos.append({
        "configuracao": {
            "faltantes": "mediana_global", "outliers": "nenhum",
            "sfh": "original", "encoding": "onehot",
            "escala": "log1p_zscore", "selecao": "nenhuma",
            "pca": "nenhum", "ponderacao": "nenhuma",
            "desbalanceamento": "nenhum",
        },
        "preprocessamento": _prep_log1p_zscore(),
    })

    return experimentos


def experimentos_selecao() -> list[dict]:
    passos_num_std = [("escala", StandardScaler())]
    mi_rs42 = partial(mutual_info_regression, random_state=42)

    variantes_selecao = [
        {"nome": "mi_50", "seletor": SelectPercentile(mi_rs42, percentile=50)},
        {"nome": "mi_25", "seletor": SelectPercentile(mi_rs42, percentile=25)},
        {"nome": "f_50", "seletor": SelectPercentile(f_regression, percentile=50)},
        {"nome": "spearman_085_005", "seletor": SpearmanSelector(limiar_par=0.85, limiar_alvo=0.05)},
        {"nome": "variancia", "seletor": VarianceThreshold(threshold=0.01)},
    ]

    experimentos = []
    for v in variantes_selecao:
        experimentos.append({
            "configuracao": {
                "faltantes": "mediana_global", "outliers": "nenhum",
                "sfh": "original", "encoding": "onehot",
                "escala": "standard", "selecao": v["nome"],
                "pca": "nenhum", "ponderacao": "nenhuma",
                "desbalanceamento": "nenhum",
            },
            "preprocessamento": _prep_escala(
                passos_num_std,
                passos_pos=[("selecao", v["seletor"])],
            ),
        })

    return experimentos


def experimentos_pca() -> list[dict]:
    passos_num_std = [("escala", StandardScaler())]

    variantes_pca = [
        {"nome": "pca_90", "variancia": 0.90},
        {"nome": "pca_95", "variancia": 0.95},
        {"nome": "pca_99", "variancia": 0.99},
    ]

    experimentos = []
    for v in variantes_pca:
        experimentos.append({
            "configuracao": {
                "faltantes": "mediana_global", "outliers": "nenhum",
                "sfh": "original", "encoding": "onehot",
                "escala": "standard", "selecao": "nenhuma",
                "pca": v["nome"], "ponderacao": "nenhuma",
                "desbalanceamento": "nenhum",
            },
            "preprocessamento": _prep_escala(
                passos_num_std,
                passos_pos=[("pca", PCA(n_components=v["variancia"], random_state=42))],
            ),
        })

    return experimentos


def experimentos_ponderacao() -> list[dict]:
    passos_num_std = [("escala", StandardScaler())]

    variantes_pond = [
        {"nome": "pond_spearman", "ponderador": PonderacaoSpearman()},
        {"nome": "pond_mi", "ponderador": PonderacaoMI(random_state=42)},
    ]

    experimentos = []
    for v in variantes_pond:
        experimentos.append({
            "configuracao": {
                "faltantes": "mediana_global", "outliers": "nenhum",
                "sfh": "original", "encoding": "onehot",
                "escala": "standard", "selecao": "nenhuma",
                "pca": "nenhum", "ponderacao": v["nome"],
                "desbalanceamento": "nenhum",
            },
            "preprocessamento": _prep_escala(
                passos_num_std,
                passos_pos=[("ponderacao", v["ponderador"])],
            ),
        })

    return experimentos


def experimentos_f3() -> list[dict]:
    return (
        experimentos_escala()
        + experimentos_selecao()
        + experimentos_pca()
        + experimentos_ponderacao()
    )


if __name__ == "__main__":
    import time
    from sklearn.base import clone
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import KFold
    from sklearn.neighbors import KNeighborsRegressor
    from baseline import ALVO, carregar_e_limpar

    print("=== Teste rapido de escala_pca_selecao.py (2 folds) ===\n")

    df = carregar_e_limpar()
    X = df[NUMERICAS + CATEGORICAS]
    y = df[ALVO]

    cv = KFold(n_splits=2, shuffle=True, random_state=42)
    knn = KNeighborsRegressor(n_neighbors=5, weights="uniform", metric="minkowski", p=2)

    todos = experimentos_f3()
    print(f"Total de experimentos F3: {len(todos)}\n")

    for exp in todos:
        cfg = exp["configuracao"]
        prep = exp["preprocessamento"]
        nome = f"escala={cfg['escala']} sel={cfg['selecao']} pca={cfg['pca']} pond={cfg['ponderacao']}"
        maes = []
        t0 = time.perf_counter()

        for treino_idx, teste_idx in cv.split(X):
            X_tr, X_te = X.iloc[treino_idx], X.iloc[teste_idx]
            y_tr, y_te = y.iloc[treino_idx], y.iloc[teste_idx]

            p = clone(prep)
            X_tr_t = p.fit_transform(X_tr, np.log1p(y_tr))
            X_te_t = p.transform(X_te)

            m = clone(knn)
            m.fit(X_tr_t, np.log1p(y_tr))
            pred = np.expm1(m.predict(X_te_t))
            maes.append(mean_absolute_error(y_te, pred))

        elapsed = time.perf_counter() - t0
        n_col = X_tr_t.shape[1]
        print(f"  {nome}")
        print(f"    n_colunas={n_col} | MAE_medio={np.mean(maes):,.0f} | tempo={elapsed:.1f}s")

    print("\n=== Teste concluido ===")
