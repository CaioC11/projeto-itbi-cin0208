
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


# --------------------------------------------------
# METRICAS
# --------------------------------------------------

def calcular_metricas(y_real, y_pred):
    return {
        "mae": mean_absolute_error(y_real, y_pred),
        "mape": mean_absolute_percentage_error(y_real, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_real, y_pred)),
        "r2": r2_score(y_real, y_pred),
    }


# --------------------------------------------------
# IDENTIFICACAO DO EXPERIMENTO
# --------------------------------------------------

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


# --------------------------------------------------
# SALVAR RESULTADOS
# --------------------------------------------------

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


# --------------------------------------------------
# EXECUTOR PRINCIPAL
# --------------------------------------------------

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

    # Remocao de linhas com faltantes antes do CV.
    if conjunto == "sem_faltantes":
        mascara = X.notna().all(axis=1) & y.notna()

        X = X.loc[mascara]
        y = y.loc[mascara]

    if len(X) < 10:
        raise ValueError(
            "Quantidade insuficiente de registros para 10 folds"
        )

    if alvo_log and (y <= -1).any():
        raise ValueError(
            "O alvo precisa ser maior que -1 para log1p"
        )

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

    tempos_fit = []
    tempos_pred = []
    quantidades_colunas = []

    # --------------------------------------------------
    # VALIDACAO CRUZADA
    # --------------------------------------------------

    for treino_idx, teste_idx in cv.split(X):

        inicio_fit = time.perf_counter()

        X_treino = X.iloc[treino_idx].copy()
        X_teste = X.iloc[teste_idx].copy()

        y_treino = y.iloc[treino_idx].copy()
        y_teste = y.iloc[teste_idx].to_numpy()

        # 1. FILTRO: somente no treino.

        if filtro_treino is not None:
            X_treino, y_treino = filtro_treino(
                X_treino,
                y_treino,
            )

        y_treino = np.asarray(y_treino).ravel()

        if len(X_treino) != len(y_treino):
            raise ValueError(
                "Filtro retornou quantidades diferentes de X e y"
            )

        if len(y_treino) < 5:
            raise ValueError(
                "Menos de 5 registros disponiveis no treino"
            )

        # 2. TRANSFORMACAO DO ALVO.

        if alvo_log:
            if np.any(y_treino <= -1):
                raise ValueError(
                    "O alvo do treino precisa ser maior que -1"
                )

            y_ajuste = np.log1p(y_treino)

        else:
            y_ajuste = y_treino.copy()

        # 3. PRE-PROCESSAMENTO.
        # Ajustado exclusivamente no treino de cada fold.

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

        # 4. REAMOSTRAGEM: somente no treino transformado.

        if reamostrar is not None:
            X_treino_transformado, y_ajuste = reamostrar(
                X_treino_transformado,
                y_ajuste,
            )

        y_ajuste = np.asarray(y_ajuste).ravel()

        if X_treino_transformado.shape[0] != len(y_ajuste):
            raise ValueError(
                "Reamostragem retornou quantidades diferentes de X e y"
            )

        if len(y_ajuste) < 5:
            raise ValueError(
                "Reamostragem deixou menos de 5 registros"
            )

        # 5. TREINAMENTO DO KNN PADRAO.

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

        # 6. PREDICAO: teste permanece intacto.

        inicio_pred = time.perf_counter()

        predicao_ajuste = knn.predict(
            X_teste_transformado
        )

        tempos_pred.append(
            time.perf_counter() - inicio_pred
        )

        # 7. RETORNO DO ALVO PARA REAIS.

        if alvo_log:
            predicao = np.expm1(predicao_ajuste)
        else:
            predicao = predicao_ajuste

        # 8. METRICAS EM REAIS.

        metricas = calcular_metricas(
            y_teste,
            predicao,
        )

        for nome, valor in metricas.items():
            resultados[nome].append(float(valor))

        # 9. METRICAS NO ESPACO LOG.

        if alvo_log:
            predicao_log = predicao_ajuste
        else:
            predicao_log = np.log1p(
                np.maximum(predicao, 0)
            )

        metricas_log = calcular_metricas(
            np.log1p(y_teste),
            predicao_log,
        )

        for nome, valor in metricas_log.items():
            resultados_log[nome].append(float(valor))

    # --------------------------------------------------
    # MONTAGEM DO RESULTADO
    # --------------------------------------------------

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

        # MAE individual dos 10 folds.
        # Necessario para o teste de Wilcoxon.

        "mae_por_fold": json.dumps(
            resultados["mae"]
        ),

        "mae_log_por_fold": json.dumps(
            resultados_log["mae"]
        ),
    }

    # Media e desvio-padrao das metricas.

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

    if salvar:
        salvar_resultado(
            resultado,
            saida,
        )

    return resultado


# --------------------------------------------------
# DIVISAO DOS EXPERIMENTOS ENTRE MAQUINAS
# --------------------------------------------------

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

        print(
            f"Experimento: {resultado['id']}"
        )

        print(
            f"MAE: {resultado['mae_media']:.3f}"
        )

        print(
            f"R2: {resultado['r2_media']:.3f}"
        )

    return resultados


# --------------------------------------------------
# VALIDACAO DO ARGUMENTO --PARTE
# --------------------------------------------------

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


# --------------------------------------------------
# EXECUCAO PELO TERMINAL
# --------------------------------------------------

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

    # Baseline B0 para validacao do executor.
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
