import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from baseline import ALVO, CATEGORICAS, NUMERICAS, carregar_e_limpar, pre_processamento
from reamostragem import (
    ruido_gaussiano,
    random_over,
    random_under,
    smogn,
    smoter,
)

N_FOLDS = 10


def _para_denso(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X, dtype=float)


def carregar_fold_real():
    df = carregar_e_limpar()
    X = df[NUMERICAS + CATEGORICAS].copy()
    y = df[ALVO].copy()

    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    treino_idx, _ = next(cv.split(X))

    X_treino = X.iloc[treino_idx]
    y_log = np.log1p(y.iloc[treino_idx].to_numpy())

    prep = pre_processamento(True)
    X_proc = _para_denso(prep.fit_transform(X_treino, y_log))
    return X_proc, y_log


def medir(nome, funcao, X, y):
    inicio = time.perf_counter()
    _, y_novo = funcao(X, y)
    segundos = time.perf_counter() - inicio
    return {
        "variante": nome,
        "colunas_entrada": X.shape[1],
        "linhas_saida": len(y_novo),
        "seg_por_fold": round(segundos, 3),
        "seg_10_folds": round(segundos * N_FOLDS, 2),
    }


if __name__ == "__main__":
    print("Carregando e pre-processando Fold 0 real do ITBI...")
    X_real, y_real = carregar_fold_real()
    print(f"Matriz de treino Fold 0: {X_real.shape[0]} linhas x {X_real.shape[1]} colunas\n")

    variantes_full = [
        ("random_over", random_over),
        ("random_under", random_under),
        ("gaussian_noise", ruido_gaussiano),
        ("smoter", smoter),
    ]

    resultados = [medir(nome, f, X_real, y_real) for nome, f in variantes_full]

    #Medicao do SMOGN apos reducao com PCA (95% da variancia)
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_real)

    resultados.append(
        medir("smogn_0.5 (PCA 95%)", lambda X, y: smogn(X, y, rel_thres=0.5), X_pca, y_real)
    )
    resultados.append(
        medir("smogn_0.8 (PCA 95%)", lambda X, y: smogn(X, y, rel_thres=0.8), X_pca, y_real)
    )

    tabela = pd.DataFrame(resultados)
    print(tabela.to_string(index=False))