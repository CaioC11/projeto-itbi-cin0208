import argparse
import functools
import numpy as np
from sklearn.neighbors import NearestNeighbors
from baseline import pre_processamento
from executor import executar_grade
from relevancia import relevancia_cauda_direita


def denso(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X, dtype=float)


def split(y_log, limiar=0.5):
    
    #Divide as linhas em raras (nota >= limiar) e comuns (nota < limiar)
    nota = relevancia_cauda_direita(y_log, y_log)
    raros = np.where(nota >= limiar)[0]
    comuns = np.where(nota < limiar)[0]
    return raros, comuns


def random_over(X, y_log, limiar=0.5, fator=2, seed=42):
    
    #Repete as linhas raras (sorteadas com reposicao) ate ter `fator` vezes mais
    rng = np.random.default_rng(seed)
    y_log = np.asarray(y_log)
    raros, _ = split(y_log, limiar)
    extras = rng.choice(raros, size=len(raros) * (fator - 1), replace=True)
    idx = np.concatenate([np.arange(len(y_log)), extras])
    return X[idx], y_log[idx]


def random_under(X, y_log, limiar=0.5, fracao=0.5, seed=42):
    
    #Mantem todas as linhas raras e so uma fracao das comuns
    rng = np.random.default_rng(seed)
    y_log = np.asarray(y_log)
    raros, comuns = split(y_log, limiar)
    mantidas = rng.choice(comuns, size=int(len(comuns) * fracao), replace=False)
    idx = np.sort(np.concatenate([raros, mantidas]))
    return X[idx], y_log[idx]


def ruido_gaussiano(X, y_log, limiar=0.5, fator=2, sigma=0.05, seed=42):
    
    #Copia linhas raras e soma um ruido gaussiano pequeno nos atributos
    rng = np.random.default_rng(seed)
    X = denso(X)
    y_log = np.asarray(y_log)
    raros, _ = split(y_log, limiar)
    escolhidos = rng.choice(raros, size=len(raros) * (fator - 1), replace=True)
    desvio = X.std(axis=0)
    ruido = rng.normal(0, sigma * desvio, size=(len(escolhidos), X.shape[1]))
    X_novo = X[escolhidos] + ruido
    return np.vstack([X, X_novo]), np.concatenate([y_log, y_log[escolhidos]])


def smoter(X, y_log, limiar=0.5, fator=2, k=5, seed=42):
    
    #Cria linhas novas entre pares de linhas raras vizinhas (X e y interpolados)
    rng = np.random.default_rng(seed)
    X = denso(X)
    y_log = np.asarray(y_log)
    raros, _ = split(y_log, limiar)
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


def smogn(X, y_log, rel_thres=0.5, k=5, pert=0.02):
    
    #Adaptador do smogn: X (matriz) e y_log entram, X e y_log reamostrados saem
    import pandas as pd
    import smogn

    X = denso(X)
    y_log = np.asarray(y_log, dtype=float)
    colunas = [f"x{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=colunas)
    df["y"] = y_log

    p75, p95 = np.percentile(y_log, [75, 95])
    pontos = [[y_log.min(), 0, 0], [p75, 0, 0], [p95, 1, 0], [y_log.max(), 1, 0]]

    saida = smogn.smoter(
        data=df,
        y="y",
        k=k,
        pert=pert,
        samp_method="balance",
        rel_thres=rel_thres,
        rel_method="manual",
        rel_ctrl_pts_rg=pontos,
    )
    return saida[colunas].to_numpy(), saida["y"].to_numpy()


def montar_experimentos(incluir_smogn=False):
    config_base = {
        "faltantes": "mediana_global",
        "outliers": "nenhum",
        "sfh": "original",
        "encoding": "onehot",
        "escala": "standard",
        "selecao": "nenhuma",
        "pca": "nenhum",
        "ponderacao": "nenhuma",
    }

    variantes = [
        ("random_over", random_over),
        ("random_under", random_under),
        ("gaussian_noise", ruido_gaussiano),
        ("smoter", smoter),
    ]

    if incluir_smogn:
        variantes.extend([
            ("smogn_0.5", functools.partial(smogn, rel_thres=0.5)),
            ("smogn_0.8", functools.partial(smogn, rel_thres=0.8)),
        ])

    experimentos = []
    for nome, funcao in variantes:
        cfg = dict(config_base)
        cfg["desbalanceamento"] = nome
        experimentos.append({
            "configuracao": cfg,
            "preprocessamento": pre_processamento(True),
            "reamostrar": funcao,
        })

    return experimentos


def rodar_testes():
    rng = np.random.default_rng(0)
    y = 10 + rng.gamma(shape=2, scale=1, size=1000)
    X = rng.normal(size=(1000, 3))
    raros, comuns = split(y)

    Xs, ys = random_over(X, y)
    assert len(ys) == len(y) + len(raros)
    assert Xs.shape[0] == len(ys)

    Xu, yu = random_under(X, y)
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

    for corte in (0.5, 0.8):
        Xg, yg = smogn(X, y, rel_thres=corte)
        assert Xg.shape[1] == X.shape[1] and Xg.shape[0] == len(yg)
        assert not np.isnan(Xg).any() and not np.isnan(yg).any()
        print(f"smogn {corte}:", len(yg))

    print("reamostragem ok")
    print("original:", len(y), "| raras:", len(raros))
    print("sobre:", len(ys), "| sub:", len(yu))
    print("ruido:", len(yr))
    print("smoter:", len(ym))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Variantes de reamostragem F4-3")
    parser.add_argument("--testar", action="store_true", help="Roda apenas testes unitarios")
    parser.add_argument("--incluir-smogn", action="store_true", help="Inclui smogn_0.5 e smogn_0.8 (lento em alta dimensao)")
    parser.add_argument("--sem-salvar", action="store_true", help="Roda no executor sem gravar no CSV")
    args = parser.parse_args()

    if args.testar:
        rodar_testes()
    else:
        experimentos = montar_experimentos(incluir_smogn=args.incluir_smogn)
        executar_grade(experimentos, salvar=not args.sem_salvar)