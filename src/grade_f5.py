
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)

from baseline import NUMERICAS, CATEGORICAS, RAIZ
from encoding_f2 import FrequencyEncoding
from escala_pca_selecao import PonderacaoSpearman, PonderacaoMI
from executor import (
    executar_experimento,
    selecionar_parte,
    criar_chave,
)


PASTA_RESULTADOS = RAIZ / "resultados"

# Candidatos provisórios.
# Confirmar e atualizar conforme as decisões oficiais do G1.

FALTANTES = [
    "mediana_global",
    "mediana_bairro",
    "media",
]

OUTLIERS = [
    "nenhum",
    "iqr_1_5",
]

ENCODINGS = [
    "onehot",
    "ordinal",
    "frequency",
    "target_suave_10",
]

ESCALAS = [
    "standard",
    "robust",
]

PONDERACOES = [
    "nenhuma",
    "pond_spearman",
    "pond_mi",
]


class ImputacaoPorBairro(BaseEstimator, TransformerMixin):

    def __init__(self, colunas):
        self.colunas = colunas

    def fit(self, X, y=None):
        self.globais_ = X[self.colunas].median()
        self.grupos_ = {}

        for coluna in self.colunas:
            self.grupos_[coluna] = (
                X.groupby("bairro")[coluna].median()
            )

        return self

    def transform(self, X):
        X = X.copy()

        for coluna in self.colunas:
            medias_bairro = X["bairro"].map(
                self.grupos_[coluna]
            )

            X[coluna] = (
                X[coluna]
                .fillna(medias_bairro)
                .fillna(self.globais_[coluna])
            )

        return X


def criar_filtro_outliers(metodo):

    if metodo == "nenhum":
        return None

    if metodo != "iqr_1_5":
        raise ValueError(
            f"Metodo de outlier desconhecido: {metodo}"
        )

    def filtrar(X, y):
        valores = np.log1p(np.asarray(y))

        q1, q3 = np.quantile(
            valores, [0.25, 0.75]
        )

        iqr = q3 - q1

        inferior = q1 - 1.5 * iqr
        superior = q3 + 1.5 * iqr

        mascara = (
            (valores >= inferior)
            & (valores <= superior)
        )

        indices = np.flatnonzero(mascara)

        return X.iloc[indices], y.iloc[indices]

    return filtrar


def criar_encoding(metodo):

    if metodo == "onehot":
        return [
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAS,
            )
        ]

    if metodo == "ordinal":
        return [
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
                CATEGORICAS,
            )
        ]

    outras = [
        coluna
        for coluna in CATEGORICAS
        if coluna != "bairro"
    ]

    if metodo == "frequency":
        return [
            (
                "bairro",
                FrequencyEncoding(["bairro"]),
                ["bairro"],
            ),
            (
                "outras",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                outras,
            ),
        ]

    if metodo == "target_suave_10":
        return [
            (
                "bairro",
                TargetEncoder(
                    target_type="continuous",
                    smooth=10.0,
                    cv=5,
                    shuffle=True,
                    random_state=42,
                ),
                ["bairro"],
            ),
            (
                "outras",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                outras,
            ),
        ]

    raise ValueError(
        f"Encoding desconhecido: {metodo}"
    )


def montar_pipeline(config):

    faltantes = config["faltantes"]

    if faltantes == "mediana_bairro":
        tratamento = ImputacaoPorBairro(NUMERICAS)
        estrategia = "median"

    elif faltantes == "mediana_global":
        tratamento = "passthrough"
        estrategia = "median"

    elif faltantes == "media":
        tratamento = "passthrough"
        estrategia = "mean"

    else:
        raise ValueError(
            f"Tratamento de faltantes invalido: {faltantes}"
        )

    escala = config["escala"]

    if escala == "standard":
        escalador = StandardScaler()

    elif escala == "robust":
        escalador = RobustScaler()

    else:
        raise ValueError(
            f"Escala desconhecida: {escala}"
        )

    numerico = Pipeline([
        (
            "imputacao",
            SimpleImputer(strategy=estrategia),
        ),
        ("escala", escalador),
    ])

    transformadores = [
        ("num", numerico, NUMERICAS),
        *criar_encoding(config["encoding"]),
    ]

    preprocessamento = ColumnTransformer(
        transformadores,
        sparse_threshold=0,
    )

    ponderacao = config["ponderacao"]

    if ponderacao == "nenhuma":
        passo_ponderacao = "passthrough"

    elif ponderacao == "pond_spearman":
        passo_ponderacao = PonderacaoSpearman()

    elif ponderacao == "pond_mi":
        passo_ponderacao = PonderacaoMI(
            random_state=42
        )

    else:
        raise ValueError(
            f"Ponderacao desconhecida: {ponderacao}"
        )

    return Pipeline([
        ("faltantes", tratamento),
        ("encoding_escala", preprocessamento),
        ("ponderacao", passo_ponderacao),
    ])


def criar_configuracoes():
    configuracoes = []

    combinacoes = itertools.product(
        FALTANTES,
        OUTLIERS,
        ENCODINGS,
        ESCALAS,
        PONDERACOES,
    )

    for faltantes, outliers, encoding, escala, ponderacao in combinacoes:

        config = {
            "faltantes": faltantes,
            "outliers": outliers,
            "sfh": "original",
            "encoding": encoding,
            "escala": escala,
            "selecao": "nenhuma",
            "pca": "nenhum",
            "ponderacao": ponderacao,
            "desbalanceamento": "nenhum",
        }

        configuracoes.append(config)

    return configuracoes


def chave_configuracao(config):
    return criar_chave(
        config,
        conjunto="completo",
        alvo_log=True,
    )


def validar_grade(configuracoes):

    chaves = [
        chave_configuracao(c)
        for c in configuracoes
    ]

    if len(configuracoes) != 144:
        raise ValueError(
            "A grade deve conter 144 configuracoes"
        )

    if len(set(chaves)) != 144:
        raise ValueError(
            "Existem configuracoes duplicadas"
        )

    for config in configuracoes:
        montar_pipeline(config)
        criar_filtro_outliers(
            config["outliers"]
        )

    print("Configuracoes:", len(configuracoes))
    print("Chaves unicas:", len(set(chaves)))
    print("Pipelines construidos com sucesso")


def salvar_manifesto(configuracoes, caminho):

    tabela = pd.DataFrame(configuracoes)

    tabela.insert(
        0,
        "chave",
        [
            chave_configuracao(c)
            for c in configuracoes
        ],
    )

    caminho = Path(caminho)
    caminho.parent.mkdir(
        parents=True, exist_ok=True
    )

    tabela.to_csv(caminho, index=False)

    print("Manifesto:", caminho)


def chaves_existentes(saida):

    saida = Path(saida)

    if not saida.exists():
        return set()

    df = pd.read_csv(saida)

    if "chave" not in df.columns:
        return set()

    return set(df["chave"].dropna())


def executar_configuracoes(
    configuracoes,
    saida,
    salvar=True,
):

    concluidas = (
        chaves_existentes(saida)
        if salvar
        else set()
    )

    resultados = []

    for indice, config in enumerate(configuracoes, 1):

        chave = chave_configuracao(config)

        if chave in concluidas:
            print(
                f"[{indice}] Experimento ja registrado"
            )
            continue

        print(
            f"\n[{indice}/{len(configuracoes)}] "
            f"{config['faltantes']} / "
            f"{config['outliers']} / "
            f"{config['encoding']} / "
            f"{config['escala']} / "
            f"{config['ponderacao']}",
            flush=True,
        )

        resultado = executar_experimento(
            configuracao=config,
            preprocessamento=montar_pipeline(config),
            filtro_treino=criar_filtro_outliers(
                config["outliers"]
            ),
            conjunto="completo",
            alvo_log=True,
            salvar=salvar,
            saida=saida,
        )

        resultados.append(resultado)
        concluidas.add(chave)

        print(
            f"MAE: {resultado['mae_media']:.2f}",
            flush=True,
        )

    return resultados


def main():

    parser = argparse.ArgumentParser(
        description="Grade combinada F5-4"
    )

    parser.add_argument(
        "--parte",
        default=None,
    )

    parser.add_argument(
        "--executar",
        action="store_true",
    )

    parser.add_argument(
        "--confirmar-g1",
        action="store_true",
        help="Confirma que os candidatos foram aprovados pelo grupo",
    )

    parser.add_argument(
        "--teste",
        action="store_true",
        help="Executa somente uma configuracao sem salvar",
    )

    parser.add_argument(
        "--saida",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    configuracoes = criar_configuracoes()

    validar_grade(configuracoes)

    selecionadas = selecionar_parte(
        configuracoes,
        args.parte,
    )

    print(
        "Configuracoes selecionadas:",
        len(selecionadas),
    )

    if args.teste:
        if not selecionadas:
            print("Nenhuma configuracao nesta parte")
            return

        executar_configuracoes(
            selecionadas[:1],
            saida=PASTA_RESULTADOS / "teste_f5.csv",
            salvar=False,
        )
        return

    if args.executar:

        if not args.confirmar_g1:
            raise SystemExit(
                "Execucao oficial bloqueada: "
                "confirme os vencedores do G1 antes."
            )

        if args.parte is None:
            raise SystemExit(
                "Informe --parte i/4 para executar a grade."
            )

        if args.saida is None:
            parte = args.parte.replace("/", "_")
            saida = (
                PASTA_RESULTADOS
                / f"f5_parte_{parte}.csv"
            )
        else:
            saida = args.saida

        executar_configuracoes(
            selecionadas,
            saida=saida,
            salvar=True,
        )

    else:
        saida = (
            args.saida
            or PASTA_RESULTADOS
            / "grade_f5_configuracoes.csv"
        )

        salvar_manifesto(
            selecionadas,
            saida,
        )

        print(
            "Manifesto gerado. Nenhum modelo executado."
        )


if __name__ == "__main__":
    main()
