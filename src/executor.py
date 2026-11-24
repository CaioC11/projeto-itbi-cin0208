import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
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


SAIDA = RAIZ / "resultados" / "experimentos.csv"

METRICAS = ("mae", "mape", "rmse", "r2")
FAIXAS = ("q1", "q2", "q3", "q4", "p99")

CAMPOS_CONFIGURACAO = (
    "faltantes",
    "outliers",
    "sfh",
    "encoding",
    "escala",
    "selecao",
    "pca",
    "ponderacao",
    "desbalanceamento",
)


def calcular_metricas(y_real, y_pred):
    return {
        "mae": mean_absolute_error(y_real, y_pred),
        "mape": mean_absolute_percentage_error(y_real, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_real, y_pred)),
        "r2": r2_score(y_real, y_pred),
    }


def calcular_metricas_faixa(y_real, y_pred):
    y_real = np.asarray(y_real)
    y_pred = np.asarray(y_pred)

    if len(y_real) == 0:
        return {
            "mae": np.nan,
            "vies": np.nan,
        }

    return {
        "mae": float(np.mean(np.abs(y_pred - y_real))),
        "vies": float(np.mean(y_pred - y_real)),
    }


def calcular_limites_faixas(y_treino):
    y_treino = np.asarray(y_treino)

    return np.quantile(
        y_treino,
        [0.25, 0.50, 0.75, 0.99],
    )


def criar_mascaras_faixas(y_teste, limites):
    q25, q50, q75, p99 = limites

    y_teste = np.asarray(y_teste)

    return {
        "q1": y_teste <= q25,
        "q2": (y_teste > q25) & (y_teste <= q50),
        "q3": (y_teste > q50) & (y_teste <= q75),
        "q4": y_teste > q75,
        "p99": y_teste >= p99,
    }


def calcular_faixas(
    y_real,
    y_pred,
    y_real_log,
    y_pred_log,
    limites,
):
    mascaras = criar_mascaras_faixas(
        y_real,
        limites,
    )

    resultado = {}

    for faixa, mascara in mascaras.items():
        metricas = calcular_metricas_faixa(
            y_real[mascara],
            y_pred[mascara],
        )

        metricas_log = calcular_metricas_faixa(
            y_real_log[mascara],
            y_pred_log[mascara],
        )

        resultado[faixa] = {
            "mae": metricas["mae"],
            "vies": metricas["vies"],
            "mae_log": metricas_log["mae"],
            "vies_log": metricas_log["vies"],
            "n": int(np.sum(mascara)),
        }

    return resultado


def resumir_valores(valores):
    valores = np.asarray(valores, dtype=float)
    validos = valores[np.isfinite(valores)]

    if len(validos) == 0:
        return float("nan"), float("nan")

    return (
        float(np.mean(validos)),
        float(np.std(validos)),
    )


def criar_chave(configuracao, conjunto, alvo_log):
    dados = {
        "configuracao": configuracao,
        "conjunto": conjunto,
        "alvo_log": alvo_log,
    }

    return json.dumps(
        dados,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )


def salvar_resultado(resultado, saida=SAIDA):
    saida = Path(saida)
    saida.parent.mkdir(parents=True, exist_ok=True)

    novo = pd.DataFrame([resultado])

    if saida.exists():
        anterior = pd.read_csv(saida)

        if "chave" in anterior.columns:
            if resultado["chave"] in anterior["chave"].astype(str).values:
                raise ValueError(
                    "Experimento ja registrado no CSV"
                )

        colunas = list(
            dict.fromkeys(
                list(anterior.columns) + list(novo.columns)
            )
        )

        anterior = anterior.reindex(columns=colunas)
        novo = novo.reindex(columns=colunas)

        tabela = pd.concat(
            [anterior, novo],
            ignore_index=True,
        )

        tabela.to_csv(saida, index=False)

    else:
        novo.to_csv(
            saida,
            index=False,
            encoding="utf-8",
        )


def executar_experimento(
    configuracao,
    preprocessamento,
    conjunto="completo",
    filtro_treino=None,
    reamostrar=None,
    alvo_log=True,
    salvar=True,
    saida=SAIDA,
    dados=None,
):
    if conjunto not in ("completo", "sem_faltantes"):
        raise ValueError("Conjunto invalido")

    df = (
        carregar_e_limpar()
        if dados is None
        else dados.copy()
    )

    colunas = NUMERICAS + CATEGORICAS

    X = df[colunas].copy()
    y = df[ALVO].copy()

    if conjunto == "sem_faltantes":
        mascara = X.notna().all(axis=1) & y.notna()

        X = X.loc[mascara]
        y = y.loc[mascara]

    if len(X) < 10:
        raise ValueError(
            "Quantidade insuficiente para 10 folds"
        )

    if y.isna().any():
        raise ValueError("O alvo contem valores ausentes")

    if (y <= -1).any():
        raise ValueError(
            "O alvo precisa ser maior que -1"
        )

    # Identifica os dados de entrada efetivamente usados na validacao.
    # O hash inclui valores, indices e ordem das colunas.
    dados_hash_obj = hashlib.sha256()
    dados_hash_obj.update(
        pd.util.hash_pandas_object(X, index=True).to_numpy().tobytes()
    )
    dados_hash_obj.update(
        pd.util.hash_pandas_object(y, index=True).to_numpy().tobytes()
    )
    dados_hash_obj.update(
        json.dumps(list(X.columns), ensure_ascii=False).encode("utf-8")
    )
    dados_hash = dados_hash_obj.hexdigest()

    protocolo_cv = "KFold_10_shuffle_True_seed_42"
    folds_hash_obj = hashlib.sha256()

    cv = KFold(
        n_splits=10,
        shuffle=True,
        random_state=42,
    )

    resultados = {
        nome: [] for nome in METRICAS
    }

    resultados_log = {
        nome: [] for nome in METRICAS
    }

    resultados_faixas = {
        faixa: {
            "mae": [],
            "vies": [],
            "mae_log": [],
            "vies_log": [],
            "n": [],
        }
        for faixa in FAIXAS
    }

    tempos_fit = []
    tempos_pred = []
    quantidades_colunas = []

    for treino_idx, teste_idx in cv.split(X):
        # Mesma ordem de indices de teste em cada fold => mesmo hash.
        folds_hash_obj.update(
            np.asarray(teste_idx, dtype=np.int64).tobytes()
        )
        inicio_fit = time.perf_counter()

        X_treino = X.iloc[treino_idx].copy()
        X_teste = X.iloc[teste_idx].copy()

        y_treino = y.iloc[treino_idx].copy()
        y_teste = y.iloc[teste_idx].to_numpy()

        # Limites calculados com o treino original.
        # Nunca utilizamos os precos do teste
        # para definir os quartis ou o P99.
        limites = calcular_limites_faixas(y_treino)

        # Filtro aplicado exclusivamente no treino.
        if filtro_treino is not None:
            X_treino, y_treino = filtro_treino(
                X_treino,
                y_treino,
            )

        y_treino = np.asarray(y_treino).ravel()

        if len(X_treino) != len(y_treino):
            raise ValueError(
                "Filtro retornou tamanhos diferentes"
            )

        if len(y_treino) < 5:
            raise ValueError(
                "Menos de 5 registros no treino"
            )

        if alvo_log:
            if np.any(y_treino <= -1):
                raise ValueError(
                    "Alvo invalido para log1p"
                )

            y_ajuste = np.log1p(y_treino)
        else:
            y_ajuste = y_treino.copy()

        prep = clone(preprocessamento)

        X_treino_transformado = prep.fit_transform(
            X_treino,
            y_ajuste,
        )

        X_teste_transformado = prep.transform(
            X_teste
        )

        quantidades_colunas.append(
            X_treino_transformado.shape[1]
        )

        # Reamostragem somente no treino.
        if reamostrar is not None:
            X_treino_transformado, y_ajuste = reamostrar(
                X_treino_transformado,
                y_ajuste,
            )

        y_ajuste = np.asarray(y_ajuste).ravel()

        if X_treino_transformado.shape[0] != len(y_ajuste):
            raise ValueError(
                "Reamostragem retornou tamanhos diferentes"
            )

        if len(y_ajuste) < 5:
            raise ValueError(
                "Menos de 5 registros apos reamostragem"
            )

        knn = KNeighborsRegressor(
            n_neighbors=5,
            weights="uniform",
            metric="minkowski",
            p=2,
        )

        knn.fit(
            X_treino_transformado,
            y_ajuste,
        )

        tempos_fit.append(
            time.perf_counter() - inicio_fit
        )

        inicio_pred = time.perf_counter()

        predicao_ajuste = knn.predict(
            X_teste_transformado
        )

        tempos_pred.append(
            time.perf_counter() - inicio_pred
        )

        if alvo_log:
            predicao = np.expm1(predicao_ajuste)
            predicao_log = predicao_ajuste
        else:
            predicao = predicao_ajuste
            predicao_log = np.log1p(
                np.maximum(predicao, 0)
            )

        y_teste_log = np.log1p(y_teste)

        # Metricas globais em reais.
        metricas = calcular_metricas(
            y_teste,
            predicao,
        )

        for nome, valor in metricas.items():
            resultados[nome].append(float(valor))

        # Metricas globais em log.
        metricas_log = calcular_metricas(
            y_teste_log,
            predicao_log,
        )

        for nome, valor in metricas_log.items():
            resultados_log[nome].append(float(valor))

        # F5-2: metricas por faixa de preco.
        metricas_faixas = calcular_faixas(
            y_teste,
            predicao,
            y_teste_log,
            predicao_log,
            limites,
        )

        for faixa in FAIXAS:
            for nome, valor in metricas_faixas[faixa].items():
                resultados_faixas[faixa][nome].append(
                    valor
                )

    folds_hash = folds_hash_obj.hexdigest()

    configuracao_completa = {
        campo: "nao_aplicado"
        for campo in CAMPOS_CONFIGURACAO
    }

    configuracao_completa.update(configuracao)

    chave = criar_chave(
        configuracao_completa,
        conjunto,
        alvo_log,
    )

    identificador = hashlib.sha256(
        chave.encode("utf-8")
    ).hexdigest()[:12]

    resultado = {
        "id": identificador,
        "chave": chave,
        "dados_hash": dados_hash,
        "folds_hash": folds_hash,
        "protocolo_cv": protocolo_cv,
        "conjunto": conjunto,
        **configuracao_completa,
        "alvo_log": alvo_log,
        "n_colunas": float(
            np.mean(quantidades_colunas)
        ),
        "tempo_fit_s": float(
            np.mean(tempos_fit)
        ),
        "tempo_pred_s": float(
            np.mean(tempos_pred)
        ),
        "mae_por_fold": json.dumps(
            resultados["mae"]
        ),
        "mae_log_por_fold": json.dumps(
            resultados_log["mae"]
        ),
    }

    # Resumo das metricas globais.
    for nome in METRICAS:
        valores = np.asarray(resultados[nome])
        valores_log = np.asarray(resultados_log[nome])

        resultado[f"{nome}_media"] = float(
            np.mean(valores)
        )

        resultado[f"{nome}_desvio"] = float(
            np.std(valores)
        )

        resultado[f"{nome}_log_media"] = float(
            np.mean(valores_log)
        )

        resultado[f"{nome}_log_desvio"] = float(
            np.std(valores_log)
        )

    # F5-2: resumo das metricas por faixa.
    for faixa in FAIXAS:
        for nome in ("mae", "vies", "mae_log", "vies_log"):
            media, desvio = resumir_valores(
                resultados_faixas[faixa][nome]
            )

            resultado[f"{faixa}_{nome}_media"] = media
            resultado[f"{faixa}_{nome}_desvio"] = desvio

        resultado[f"{faixa}_n_total"] = int(
            np.sum(resultados_faixas[faixa]["n"])
        )

    if salvar:
        salvar_resultado(
            resultado,
            saida,
        )

    return resultado


def selecionar_parte(experimentos, parte=None):
    experimentos = list(experimentos)

    if parte is None:
        return experimentos

    try:
        numero, total = map(
            int,
            parte.split("/"),
        )
    except (ValueError, AttributeError):
        raise ValueError(
            "Formato esperado: --parte 1/4"
        )

    if total < 1 or numero < 1 or numero > total:
        raise ValueError("Parte invalida")

    return [
        experimento
        for indice, experimento in enumerate(experimentos)
        if indice % total == numero - 1
    ]


def executar_grade(
    experimentos,
    parte=None,
    salvar=True,
    saida=SAIDA,
):
    selecionados = selecionar_parte(
        experimentos,
        parte,
    )

    resultados = []

    for experimento in selecionados:
        parametros = dict(experimento)

        configuracao = parametros.pop("configuracao")
        prep = parametros.pop("preprocessamento")

        resultado = executar_experimento(
            configuracao=configuracao,
            preprocessamento=prep,
            salvar=salvar,
            saida=saida,
            **parametros,
        )

        resultados.append(resultado)

        print(f"Experimento: {resultado['id']}")
        print(f"MAE: {resultado['mae_media']:.3f}")
        print(f"R2: {resultado['r2_media']:.3f}")

    return resultados


def validar_parte(valor):
    try:
        numero, total = map(
            int,
            valor.split("/"),
        )
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Use o formato i/4, por exemplo 1/4"
        )

    if total < 1 or numero < 1 or numero > total:
        raise argparse.ArgumentTypeError(
            "Divisao de partes invalida"
        )

    return valor


def main():
    parser = argparse.ArgumentParser(
        description="Executor de experimentos ITBI"
    )

    parser.add_argument(
        "--parte",
        type=validar_parte,
        default=None,
        help="Parte da grade, exemplo: 1/4",
    )

    parser.add_argument(
        "--sem-salvar",
        action="store_true",
        help="Executa sem registrar no CSV",
    )

    parser.add_argument(
        "--saida",
        type=Path,
        default=SAIDA,
        help="Caminho do CSV de resultados",
    )

    args = parser.parse_args()

    # Baseline B0 para validacao.
    # A grade completa sera fornecida pela F5-4.
    experimentos = [
        {
            "configuracao": {
                "faltantes": "mediana_global",
                "outliers": "nenhum",
                "sfh": "original",
                "encoding": "onehot",
                "escala": "standard",
                "selecao": "nenhuma",
                "pca": "nenhum",
                "ponderacao": "nenhuma",
                "desbalanceamento": "nenhum",
            },
            "preprocessamento": pre_processamento(True),
        }
    ]

    selecionados = selecionar_parte(
        experimentos,
        args.parte,
    )

    print(
        f"Experimentos selecionados: {len(selecionados)}"
    )

    if not selecionados:
        print("Nenhum experimento destinado a esta parte.")
        return

    executar_grade(
        experimentos,
        parte=args.parte,
        salvar=not args.sem_salvar,
        saida=args.saida,
    )


if __name__ == "__main__":
    main()