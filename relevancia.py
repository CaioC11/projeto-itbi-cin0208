import numpy as np


def relevancia_cauda_direita(y, y_treino, p_inicio=75, p_fim=95):
    """Nota de 0 a 1 para cada valor de y (alvo em log).

    Valores até o percentil p_inicio do treino recebem 0.
    Valores a partir do percentil p_fim recebem 1.
    Entre os dois, a nota sobe em linha reta.

    Os percentis vêm só de y_treino, para não vazar dados do teste.
    """
    y = np.asarray(y, dtype=float)
    inicio, fim = np.percentile(y_treino, [p_inicio, p_fim])
    return np.clip((y - inicio) / (fim - inicio), 0, 1)

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    #Cauda a direita
    y_treino = 10 + rng.gamma(shape=2, scale=1, size=1000)

    nota = relevancia_cauda_direita(y_treino, y_treino)

    assert nota.min() >= 0 and nota.max() <= 1
    assert nota[y_treino <= np.percentile(y_treino, 75)].max() == 0
    assert nota[y_treino >= np.percentile(y_treino, 95)].min() == 1
    ordem = np.argsort(y_treino)
    assert np.all(np.diff(nota[ordem]) >= 0)  # nunca diminui quando y aumenta

    print("relevância ok")
    print("% com nota 0:", round((nota == 0).mean() * 100, 1))
    print("% com nota 1:", round((nota == 1).mean() * 100, 1))