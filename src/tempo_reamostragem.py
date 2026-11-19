import time
import numpy as np
import pandas as pd

from reamostragem import (
    sobreamostrar,
    subamostrar,
    ruido_gaussiano,
    smoter,
    smogn_reamostrar,
)
#Trocar linhas e colunas pelo real tamanho do treino quando tiver os dados. O plano fala em ~130 colunas depois do One-Hot.
N_LINHAS = 3000 
N_COLUNAS = 130
N_FOLDS = 10


def medir(nome, funcao, X, y):
    inicio = time.perf_counter()
    X_novo, y_novo = funcao(X, y)
    segundos = time.perf_counter() - inicio
    return {
        "variante": nome,
        "linhas_saida": len(y_novo),
        "seg_por_fold": round(segundos, 2),
        "seg_10_folds": round(segundos * N_FOLDS, 1),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    y = 10 + rng.gamma(shape=2, scale=1, size=N_LINHAS)
    X = rng.normal(size=(N_LINHAS, N_COLUNAS))

    variantes = [
        ("random_over", sobreamostrar),
        ("random_under", subamostrar),
        ("gaussian_noise", ruido_gaussiano),
        ("smoter", smoter),
        ("smogn_0.5", lambda X, y: smogn_reamostrar(X, y, rel_thres=0.5)),
        ("smogn_0.8", lambda X, y: smogn_reamostrar(X, y, rel_thres=0.8)),
    ]

    resultados = [medir(nome, f, X, y) for nome, f in variantes]
    tabela = pd.DataFrame(resultados)

    print()
    print(f"Dados de exemplo: {N_LINHAS} linhas x {N_COLUNAS} colunas")
    print(tabela.to_string(index=False))