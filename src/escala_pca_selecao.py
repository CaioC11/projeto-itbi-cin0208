# Escala, PCA e selecao de atributos — F3 (Eugenio)
# F3-1: sem escala | Min-Max | Robust | Yeo-Johnson | log1p+Z-score
# F3-2: info mutua 50/25% | teste F 50% | Spearman | variancia
# F3-3: PCA 90% | 95% | 99% de variancia
# F3-4: ponderacao por |Spearman| | por info mutua

from __future__ import annotations

import sys
import warnings
from pathlib import Path

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

warnings.filterwarnings("ignore")

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

NUMERICAS = [
    "ano_construcao", "area_terreno", "area_construida",
    "fracao_ideal", "sfh", "latitude", "longitude",
]
CATEGORICAS = [
    "bairro", "padrao_acabamento", "tipo_construcao",
    "tipo_ocupacao", "estado_conservacao", "tipo_imovel",
]

# colunas com assimetria positiva forte — recebem log1p antes do Z-score
COLUNAS_ASSIMETRICAS = ["area_terreno", "area_construida", "fracao_ideal", "sfh"]
COLUNAS_SIMETRICAS   = [c for c in NUMERICAS if c not in COLUNAS_ASSIMETRICAS]

class Log1pSeletivoScaler(BaseEstimator, TransformerMixin):
    # aplica log1p nas primeiras n_assimetricas colunas, depois Z-score em todas

    def __init__(self, n_assimetricas: int = len(COLUNAS_ASSIMETRICAS)):
        self.n_assimetricas = n_assimetricas

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        Xc = X.copy()
        Xc[:, : self.n_assimetricas] = np.log1p(
            np.clip(Xc[:, : self.n_assimetricas], 0, None)
        )
        self.mean_ = np.nanmean(Xc, axis=0)
        self.std_  = np.nanstd(Xc, axis=0)
        self.std_  = np.where(self.std_ == 0, 1.0, self.std_)
        return self

    def transform(self, X, y=None):
        X = np.asarray(X, dtype=float).copy()
        X[:, : self.n_assimetricas] = np.log1p(
            np.clip(X[:, : self.n_assimetricas], 0, None)
        )
        return (X - self.mean_) / self.std_

class SpearmanSelector(BaseEstimator, TransformerMixin):
    # remove atributos com |rho_alvo| < limiar_alvo
    # e, de pares com |rho_par| > limiar_par, descarta o menos correlacionado com o alvo

    def __init__(self, limiar_par: float = 0.85, limiar_alvo: float = 0.05):
        self.limiar_par  = limiar_par
        self.limiar_alvo = limiar_alvo

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n = X.shape[1]

        cor_alvo = np.array([
            abs(spearmanr(X[:, j], y, nan_policy="omit").correlation or 0.0)
            for j in range(n)
        ])

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
                if abs(r or 0.0) > self.limiar_par:
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
        self.pesos_ = np.array([
            abs(spearmanr(X[:, j], y, nan_policy="omit").correlation or 0.0)
            for j in range(n)
        ])
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
    # assimétricas chegam nos índices 0..3 porque _prep_log1p_zscore reordena as colunas
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
    # passa assimetricas primeiro para que o Log1pSeletivoScaler ache os indices certos
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
        {"nome_escala": "minmax",     "passos_num": [("escala", MinMaxScaler())]},
        {"nome_escala": "robust",     "passos_num": [("escala", RobustScaler())]},
        {
            # StandardScaler antes do PowerTransformer evita BracketError
            # quando colunas como sfh chegam na casa dos milhoes
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

    variantes_selecao = [
        {"nome": "mi_50",            "seletor": SelectPercentile(mutual_info_regression, percentile=50)},
        {"nome": "mi_25",            "seletor": SelectPercentile(mutual_info_regression, percentile=25)},
        {"nome": "f_50",             "seletor": SelectPercentile(f_regression, percentile=50)},
        {"nome": "spearman_085_005", "seletor": SpearmanSelector(limiar_par=0.85, limiar_alvo=0.05)},
        {"nome": "variancia",        "seletor": VarianceThreshold(threshold=0.01)},
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
        {"nome": "pond_mi",       "ponderador": PonderacaoMI(random_state=42)},
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
        experimentos_escala()       # 5
        + experimentos_selecao()    # 5
        + experimentos_pca()        # 3
        + experimentos_ponderacao() # 2
    )

if __name__ == "__main__":
    import time

    sys.path.insert(0, str(RAIZ / "src"))
    from baseline import carregar_e_limpar, ALVO, NUMERICAS as NUM, CATEGORICAS as CAT
    from sklearn.model_selection import KFold
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.base import clone

    print("=== Teste rapido de escala_pca_selecao.py (2 folds) ===\n")

    df = carregar_e_limpar()
    X = df[NUM + CAT]
    y = df[ALVO]

    cv  = KFold(n_splits=2, shuffle=True, random_state=42)
    knn = KNeighborsRegressor(n_neighbors=5, weights="uniform", metric="minkowski", p=2)

    todos = experimentos_f3()
    print(f"Total de experimentos F3: {len(todos)}\n")

    for exp in todos:
        cfg  = exp["configuracao"]
        prep = exp["preprocessamento"]
        nome = f"escala={cfg['escala']} sel={cfg['selecao']} pca={cfg['pca']} pond={cfg['ponderacao']}"
        maes = []
        t0   = time.perf_counter()

        for treino_idx, teste_idx in cv.split(X):
            X_tr, X_te = X.iloc[treino_idx], X.iloc[teste_idx]
            y_tr, y_te = y.iloc[treino_idx], y.iloc[teste_idx]

            p      = clone(prep)
            X_tr_t = p.fit_transform(X_tr, np.log1p(y_tr))
            X_te_t = p.transform(X_te)

            m    = clone(knn)
            m.fit(X_tr_t, np.log1p(y_tr))
            pred = np.expm1(m.predict(X_te_t))

            from sklearn.metrics import mean_absolute_error
            maes.append(mean_absolute_error(y_te, pred))

        elapsed = time.perf_counter() - t0
        n_col   = X_tr_t.shape[1]
        print(f"  {nome}")
        print(f"    n_colunas={n_col} | MAE_medio={np.mean(maes):,.0f} | tempo={elapsed:.1f}s")

    print("\n=== Teste concluido ===")