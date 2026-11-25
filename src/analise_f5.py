
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor

from baseline import (
    ALVO,
    CATEGORICAS,
    NUMERICAS,
    RAIZ,
    carregar_e_limpar,
    pre_processamento,
)


PASTA_SAIDA = RAIZ / "resultados" / "analise_f5"

ARQUIVOS_PADRAO = [
    RAIZ / "resultados" / "experimentos.csv",
    RAIZ / "resultados" / "experimentos_f1_3.csv",
    RAIZ / "resultados" / "experimentos_f3.csv",
    RAIZ / "resultados" / "experimentos_f4.csv",
    RAIZ / "resultados" / "referencias_f5_3.csv",
    RAIZ / "resultados" / "experimentos_f5.csv",
    RAIZ / "resultados" / "experimentos_f4_5.csv",
]

CAMPOS = [
    "faltantes",
    "outliers",
    "sfh",
    "encoding",
    "escala",
    "selecao",
    "pca",
    "ponderacao",
    "desbalanceamento",
    "conjunto",
    "alvo_log",
]

TECNICAS = [
    "faltantes",
    "outliers",
    "sfh",
    "encoding",
    "escala",
    "selecao",
    "pca",
    "ponderacao",
    "desbalanceamento",
]

BASELINE = {
    "faltantes": "mediana_global",
    "outliers": "nenhum",
    "sfh": "original",
    "encoding": "onehot",
    "escala": "standard",
    "selecao": "nenhuma",
    "pca": "nenhum",
    "ponderacao": "nenhuma",
    "desbalanceamento": "nenhum",
}


def carregar_resultados(arquivos):
    tabelas = []

    for arquivo in arquivos:
        arquivo = Path(arquivo)

        if not arquivo.exists():
            print("Arquivo ausente:", arquivo)
            continue

        tabela = pd.read_csv(arquivo)

        if tabela.empty:
            continue

        tabela["arquivo_origem"] = arquivo.name
        tabelas.append(tabela)

        print(
            f"{arquivo.name}: {len(tabela)} resultados"
        )

    if not tabelas:
        raise ValueError(
            "Nenhum CSV de experimentos encontrado"
        )

    df = pd.concat(
        tabelas,
        ignore_index=True,
        sort=False,
    )

    for campo in CAMPOS:
        if campo not in df.columns:
            if campo == "conjunto":
                df[campo] = "completo"
            elif campo == "alvo_log":
                df[campo] = True
            else:
                df[campo] = "nao_aplicado"

    df["alvo_log"] = (
        df["alvo_log"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )

    if df["alvo_log"].isna().any():
        raise ValueError("Valores invalidos em alvo_log")

    if "mae_media" not in df.columns:
        raise ValueError(
            "Coluna mae_media nao encontrada"
        )

    # A configuracao determina a identidade do
    # experimento, independentemente do formato
    # antigo ou novo da coluna chave.
    df["chave_canonica"] = df.apply(
        lambda linha: json.dumps(
            {
                campo: (
                    bool(linha[campo])
                    if campo == "alvo_log"
                    else str(linha[campo])
                )
                for campo in CAMPOS
            },
            sort_keys=True,
            ensure_ascii=False,
        ),
        axis=1,
    )

    total = len(df)

    # O ultimo resultado prevalece em duplicatas.
    df = df.drop_duplicates(
        subset="chave_canonica",
        keep="last",
    ).copy()

    print("Linhas carregadas:", total)
    print("Configuracoes unicas:", len(df))

    return df.reset_index(drop=True)


def obter_baseline(df, conjunto, alvo_log=True):
    mascara = (
        (df["conjunto"] == conjunto)
        & (df["alvo_log"] == alvo_log)
    )

    for campo, valor in BASELINE.items():
        mascara &= df[campo] == valor

    candidatos = df.loc[mascara]

    if candidatos.empty:
        return None

    return candidatos.iloc[-1]


def ler_folds(valor):
    if isinstance(valor, list):
        return np.asarray(valor, dtype=float)

    if pd.isna(valor):
        return None

    try:
        dados = json.loads(valor)
        valores = np.asarray(dados, dtype=float)
    except (ValueError, TypeError):
        return None

    if valores.shape != (10,):
        return None

    if not np.all(np.isfinite(valores)):
        return None

    return valores


def corrigir_holm(p_valores):
    """Ajusta p-valores pelo metodo Holm-Bonferroni."""
    p_valores = np.asarray(p_valores, dtype=float)
    if len(p_valores) == 0:
        return np.array([], dtype=float)
    if not np.all(np.isfinite(p_valores)):
        raise ValueError("P-valores invalidos")

    ordem = np.argsort(p_valores)
    ordenados = p_valores[ordem]
    fatores = np.arange(len(ordenados), 0, -1)
    corrigidos = np.minimum(
        np.maximum.accumulate(ordenados * fatores), 1.0
    )
    resultado = np.empty(len(p_valores), dtype=float)
    resultado[ordem] = corrigidos
    return resultado


def pareamento_verificado(experimento, referencia):
    """Exige proveniencia identica de dados, folds e protocolo CV."""
    for campo in ("dados_hash", "folds_hash", "protocolo_cv"):
        if campo not in experimento.index or campo not in referencia.index:
            return False
        valor_exp = experimento[campo]
        valor_ref = referencia[campo]
        if pd.isna(valor_exp) or pd.isna(valor_ref):
            return False
        if not str(valor_exp).strip() or not str(valor_ref).strip():
            return False
        if str(valor_exp) != str(valor_ref):
            return False
    return True


def teste_wilcoxon(df):
    """Compara MAEs por fold, sem afirmar significancia sem pareamento."""
    linhas = []

    for _, experimento in df.iterrows():
        referencia = obter_baseline(
            df, experimento["conjunto"], experimento["alvo_log"]
        )
        if referencia is None:
            continue
        if experimento["chave_canonica"] == referencia["chave_canonica"]:
            continue

        folds_exp = ler_folds(experimento.get("mae_por_fold"))
        folds_base = ler_folds(referencia.get("mae_por_fold"))
        if folds_exp is None or folds_base is None:
            continue

        diferencas = folds_exp - folds_base
        if np.all(diferencas == 0):
            estatistica, p_valor = 0.0, 1.0
        else:
            estatistica, p_valor = wilcoxon(
                diferencas,
                zero_method="wilcox",
                alternative="two-sided",
                method="auto",
            )

        linhas.append({
            "id": experimento.get("id"),
            "chave": experimento["chave_canonica"],
            "conjunto": experimento["conjunto"],
            "alvo_log": experimento["alvo_log"],
            "mae_experimento": experimento["mae_media"],
            "mae_baseline": referencia["mae_media"],
            "diferenca_mae": (
                experimento["mae_media"] - referencia["mae_media"]
            ),
            "estatistica": float(estatistica),
            "p_valor": float(p_valor),
            "pareamento_verificado": pareamento_verificado(
                experimento, referencia
            ),
        })

    colunas = [
        "id", "chave", "conjunto", "alvo_log", "mae_experimento",
        "mae_baseline", "diferenca_mae", "estatistica", "p_valor",
        "pareamento_verificado", "p_ajustado_holm",
        "significativo_5pct", "interpretacao",
    ]
    if not linhas:
        return pd.DataFrame(columns=colunas)

    tabela = pd.DataFrame(linhas)
    tabela["p_ajustado_holm"] = np.nan

    # Familias de comparacao: mesmo conjunto e alvo_log.
    for _, grupo in tabela.groupby(["conjunto", "alvo_log"], dropna=False):
        tabela.loc[grupo.index, "p_ajustado_holm"] = corrigir_holm(
            grupo["p_valor"].to_numpy()
        )

    tabela["significativo_5pct"] = (
        tabela["pareamento_verificado"]
        & (tabela["p_ajustado_holm"] < 0.05)
    )
    tabela["interpretacao"] = np.where(
        ~tabela["pareamento_verificado"],
        "Exploratorio: pareamento nao verificado",
        np.where(
            tabela["significativo_5pct"],
            "Diferenca significativa (Holm, 5%)",
            "Diferenca nao significativa (Holm, 5%)",
        ),
    )
    return tabela[colunas]


def tabela_ranking(df):
    colunas = [
        "id",
        "conjunto",
        "alvo_log",
        *TECNICAS,
        "mae_media",
        "mae_desvio",
        "rmse_media",
        "mape_media",
        "r2_media",
        "q4_mae_media",
    ]

    existentes = [
        c for c in colunas if c in df.columns
    ]

    return (
        df[existentes]
        .sort_values("mae_media")
        .reset_index(drop=True)
    )


def tabela_custo(df):
    colunas = [
        "id",
        "escala",
        "selecao",
        "pca",
        "ponderacao",
        "n_colunas",
        "tempo_fit_s",
        "tempo_pred_s",
    ]

    existentes = [
        c for c in colunas if c in df.columns
    ]

    tabela = df[existentes].copy()

    if {
        "tempo_fit_s",
        "tempo_pred_s",
    }.issubset(tabela.columns):
        tabela["tempo_total_s"] = (
            tabela["tempo_fit_s"]
            + tabela["tempo_pred_s"]
        )

        tabela = tabela.sort_values(
            "tempo_total_s"
        )

    return tabela


def ranking_impacto(df):
    linhas = []

    for _, experimento in df.iterrows():
        if experimento["conjunto"] != "completo":
            continue

        if not experimento["alvo_log"]:
            continue

        alteradas = [
            campo
            for campo in TECNICAS
            if str(experimento[campo])
            != BASELINE[campo]
        ]

        # Somente experimentos de um fator por vez.
        if len(alteradas) != 1:
            continue

        referencia = obter_baseline(
            df, "completo", True
        )

        if referencia is None:
            continue

        tecnica = alteradas[0]

        linhas.append({
            "tecnica": tecnica,
            "variante": experimento[tecnica],
            "mae_experimento": experimento["mae_media"],
            "mae_baseline": referencia["mae_media"],
            "melhoria_mae": (
                referencia["mae_media"]
                - experimento["mae_media"]
            ),
            "melhoria_percentual": (
                100
                * (
                    referencia["mae_media"]
                    - experimento["mae_media"]
                )
                / referencia["mae_media"]
            ),
        })

    tabela = pd.DataFrame(linhas)

    if not tabela.empty:
        tabela = tabela.sort_values(
            "melhoria_mae",
            ascending=False,
        )

    return tabela


def gerar_graficos_baseline(pasta):
    df = carregar_e_limpar()

    X = df[NUMERICAS + CATEGORICAS]
    y = df[ALVO].to_numpy()

    cv = KFold(
        n_splits=10,
        shuffle=True,
        random_state=42,
    )

    observado = np.empty(len(y), dtype=float)
    previsto = np.empty(len(y), dtype=float)

    for treino, teste in cv.split(X):
        prep = clone(pre_processamento(True))

        X_treino = prep.fit_transform(
            X.iloc[treino],
            np.log1p(y[treino]),
        )

        X_teste = prep.transform(X.iloc[teste])

        knn = KNeighborsRegressor(
            n_neighbors=5,
            weights="uniform",
            metric="minkowski",
            p=2,
        )

        knn.fit(X_treino, np.log1p(y[treino]))

        observado[teste] = y[teste]
        previsto[teste] = np.expm1(
            knn.predict(X_teste)
        )

    residuos = observado - previsto

    previsoes = pd.DataFrame({
        "observado": observado,
        "previsto": previsto,
        "residuo": residuos,
    })

    previsoes.to_csv(
        pasta / "predicoes_baseline.csv",
        index=False,
    )

    plt.figure(figsize=(8, 5))
    plt.scatter(
        observado,
        previsto,
        s=6,
        alpha=0.3,
    )

    limite = max(
        np.max(observado),
        np.max(previsto),
    )

    plt.plot(
        [0, limite],
        [0, limite],
        linestyle="--",
    )

    plt.xlabel("Valor observado (R$)")
    plt.ylabel("Valor previsto (R$)")
    plt.title("Baseline B0: previsto x observado")
    plt.tight_layout()

    plt.savefig(
        pasta / "previsto_observado_b0.png",
        dpi=150,
    )

    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(
        previsto,
        residuos,
        s=6,
        alpha=0.3,
    )

    plt.axhline(0, linestyle="--")

    plt.xlabel("Valor previsto (R$)")
    plt.ylabel("Residuo: observado - previsto (R$)")
    plt.title("Baseline B0: grafico de residuos")
    plt.tight_layout()

    plt.savefig(
        pasta / "residuos_b0.png",
        dpi=150,
    )

    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--arquivos",
        nargs="+",
        default=ARQUIVOS_PADRAO,
    )

    parser.add_argument(
        "--saida",
        type=Path,
        default=PASTA_SAIDA,
    )

    parser.add_argument(
        "--sem-graficos",
        action="store_true",
    )

    args = parser.parse_args()

    pasta = args.saida
    pasta.mkdir(parents=True, exist_ok=True)

    df = carregar_resultados(args.arquivos)

    df.to_csv(
        pasta / "todas_configuracoes.csv",
        index=False,
    )

    ranking = tabela_ranking(df)
    ranking.to_csv(
        pasta / "ranking_experimentos.csv",
        index=False,
    )

    custo = tabela_custo(df)
    custo.to_csv(
        pasta / "custo_computacional.csv",
        index=False,
    )

    impacto = ranking_impacto(df)
    impacto.to_csv(
        pasta / "impacto_tecnicas.csv",
        index=False,
    )

    wilcoxon_df = teste_wilcoxon(df)
    wilcoxon_df.to_csv(
        pasta / "wilcoxon.csv",
        index=False,
    )

    if not args.sem_graficos:
        gerar_graficos_baseline(pasta)

    print("\n=== RESUMO F5-5 ===")
    print("Experimentos unicos:", len(df))
    print("Meta minima:", 150)
    print("Faltam:", max(0, 150 - len(df)))
    print("Comparacoes Wilcoxon:", len(wilcoxon_df))

    if not ranking.empty:
        print(
            "Menor MAE registrado:",
            round(ranking.iloc[0]["mae_media"], 2),
        )

    print("Arquivos salvos em:", pasta)


if __name__ == "__main__":
    main()

