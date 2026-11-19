import numpy as np
import pandas as pd


def erro_por_faixa(y_real, y_previsto):
    
    #MAE e viés por quartil do alvo (Q1 a Q4) e para o 1% mais caro (P99).
    #Viés = média de (previsto - real): negativo quer dizer que o modelo subestima.
   
    y_real = np.asarray(y_real, dtype=float)
    y_previsto = np.asarray(y_previsto, dtype=float)
    erro = y_previsto - y_real

    q1, q2, q3, p99 = np.percentile(y_real, [25, 50, 75, 99])
    faixas = {
        "Q1": y_real <= q1,
        "Q2": (y_real > q1) & (y_real <= q2),
        "Q3": (y_real > q2) & (y_real <= q3),
        "Q4": y_real > q3,
        "P99": y_real >= p99,
    }
    linhas = []
    for nome, mascara in faixas.items():
        linhas.append({
            "faixa": nome,
            "n": int(mascara.sum()),
            "mae": np.abs(erro[mascara]).mean(),
            "vies": erro[mascara].mean(),
        })
    return pd.DataFrame(linhas)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    y_real = 10 + rng.gamma(shape=2, scale=1, size=1000)

    #Modelo de exemplo que "encolhe" as previsões em direção à média (simulação k-NN)
    y_prev = y_real.mean() + 0.7 * (y_real - y_real.mean())

    tabela = erro_por_faixa(y_real, y_prev)

    assert list(tabela["faixa"]) == ["Q1", "Q2", "Q3", "Q4", "P99"]
    assert tabela["n"].iloc[:4].sum() == len(y_real)
    assert (tabela["mae"] >= 0).all()

    #Subestima caros
    assert tabela.loc[tabela["faixa"] == "Q4", "vies"].iloc[0] < 0
    #Superestima baratos
    assert tabela.loc[tabela["faixa"] == "Q1", "vies"].iloc[0] > 0

    print("erro por faixa ok")
    print(tabela.round(3).to_string(index=False))