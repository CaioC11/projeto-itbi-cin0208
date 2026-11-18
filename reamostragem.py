import numpy as np
from relevancia import relevancia_cauda_direita
from sklearn.neighbors import NearestNeighbors


def _separar(y_log, limiar=0.5):
    
    #Divide as linhas em raras (nota >= limiar) e comuns (nota < limiar)
    nota = relevancia_cauda_direita(y_log, y_log)
    raros = np.where(nota >= limiar)[0]
    comuns = np.where(nota < limiar)[0]
    return raros, comuns


def sobreamostrar(X, y_log, limiar=0.5, fator=2, seed=42):
    
    #Repete as linhas raras (sorteadas com reposição) até ter `fator` vezes mais
    rng = np.random.default_rng(seed)
    y_log = np.asarray(y_log)
    raros, _ = _separar(y_log, limiar)
    extras = rng.choice(raros, size=len(raros) * (fator - 1), replace=True)
    idx = np.concatenate([np.arange(len(y_log)), extras])
    return X[idx], y_log[idx]


def subamostrar(X, y_log, limiar=0.5, fracao=0.5, seed=42):
    
    #Mantém todas as linhas raras e só uma fração das comuns
    rng = np.random.default_rng(seed)
    y_log = np.asarray(y_log)
    raros, comuns = _separar(y_log, limiar)
    mantidas = rng.choice(comuns, size=int(len(comuns) * fracao), replace=False)
    idx = np.sort(np.concatenate([raros, mantidas]))
    return X[idx], y_log[idx]

def ruido_gaussiano(X, y_log, limiar=0.5, fator=2, sigma=0.05, seed=42):

    #Copia linhas raras e soma um ruído gaussiano pequeno nos atributos
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    y_log = np.asarray(y_log)
    raros, _ = _separar(y_log, limiar)
    escolhidos = rng.choice(raros, size=len(raros) * (fator - 1), replace=True)
    desvio = X.std(axis=0)
    ruido = rng.normal(0, sigma * desvio, size=(len(escolhidos), X.shape[1]))
    X_novo = X[escolhidos] + ruido
    return np.vstack([X, X_novo]), np.concatenate([y_log, y_log[escolhidos]])

def smoter(X, y_log, limiar=0.5, fator=2, k=5, seed=42):
    
    #Cria linhas novas entre pares de linhas raras vizinhas (X e y interpolados)
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    y_log = np.asarray(y_log)
    raros, _ = _separar(y_log, limiar)
    k = min(k, len(raros) - 1)

    vizinhos = NearestNeighbors(n_neighbors=k + 1).fit(X[raros])
    _, ind = vizinhos.kneighbors(X[raros])

    n_novos = len(raros) * (fator - 1)
    bases = rng.integers(0, len(raros), size=n_novos)
    pares = ind[bases, rng.integers(1, k + 1, size=n_novos)]
    u = rng.random(n_novos)

    Xa, Xb = X[raros[bases]], X[raros[pares]]
    ya, yb = y_log[raros[bases]], y_log[raros[pares]]
    X_novo = Xa + u[:, None] * (Xb - Xa)
    y_novo = ya + u * (yb - ya)
    return np.vstack([X, X_novo]), np.concatenate([y_log, y_novo])


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y = 10 + rng.gamma(shape=2, scale=1, size=1000)
    X = rng.normal(size=(1000, 3))
    raros, comuns = _separar(y)

    Xs, ys = sobreamostrar(X, y)
    assert len(ys) == len(y) + len(raros)
    assert Xs.shape[0] == len(ys)

    Xu, yu = subamostrar(X, y)
    assert len(yu) == len(raros) + int(len(comuns) * 0.5)
    assert Xu.shape[0] == len(yu)

    Xr, yr = ruido_gaussiano(X, y)
    assert len(yr) == len(y) + len(raros)
    assert Xr.shape == (len(yr), X.shape[1])
    assert not np.allclose(Xr[len(y):], X[np.argsort(y)[-len(raros):]])

    Xm, ym = smoter(X, y)
    assert len(ym) == len(y) + len(raros)
    assert Xm.shape == (len(ym), X.shape[1])
    novos = ym[len(y):]
    assert novos.min() >= y[raros].min() and novos.max() <= y[raros].max()

    print("reamostragem ok")
    print("original:", len(y), "| raras:", len(raros))
    print("sobre:", len(ys), "| sub:", len(yu))
    print("ruído:", len(yr))
    print("smoter:", len(ym))