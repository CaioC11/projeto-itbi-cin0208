
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
from src.limpeza import ALVO, carregar_e_limpar

SEED = 42
SAIDA = RAIZ / "resultados" / "baseline.csv"

NUMERICAS = ["ano_construcao", "area_terreno", "area_construida",
             "fracao_ideal", "sfh", "latitude", "longitude"]
CATEGORICAS = ["bairro", "padrao_acabamento", "tipo_construcao",
               "tipo_ocupacao", "estado_conservacao", "tipo_imovel"]


def pre_processamento(escalar):
    passos = [("imputacao", SimpleImputer(strategy="median"))]
    if escalar:
        passos.append(("escala", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(passos), NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
    ])


def modelo(regressor, escalar):
    pipe = Pipeline([("prep", pre_processamento(escalar)), ("reg", regressor)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p,
                                      inverse_func=np.expm1)


def main():
    df = carregar_e_limpar()
    X, y = df[NUMERICAS + CATEGORICAS], df[ALVO]
    print(f"Instâncias após a limpeza: {len(df)} | scikit-learn {sklearn.__version__}")

    cv = KFold(n_splits=10, shuffle=True, random_state=SEED)
    scoring = {"mae": "neg_mean_absolute_error",
               "rmse": "neg_root_mean_squared_error",
               "mape": "neg_mean_absolute_percentage_error",
               "r2": "r2"}
    candidatos = {
        "Média (em log)": modelo(DummyRegressor(), False),
        "kNN k=5 sem escala": modelo(KNeighborsRegressor(), False),
        "kNN k=5 com escala": modelo(KNeighborsRegressor(), True),
    }

    linhas = []
    for nome, m in candidatos.items():
        r = cross_validate(m, X, y, cv=cv, scoring=scoring)
        linha = {"modelo": nome}
        for k in scoring:
            v = r[f"test_{k}"]
            sinal = 1 if k == "r2" else -1
            linha[f"{k}_media"] = sinal * v.mean()
            linha[f"{k}_desvio"] = v.std()
        linhas.append(linha)

    tabela = pd.DataFrame(linhas)
    SAIDA.parent.mkdir(exist_ok=True)
    tabela.round(3).to_csv(SAIDA, index=False)
    print(tabela.round(3).to_string(index=False))
    print(f"\nSalvo em {SAIDA}")


if __name__ == "__main__":
    main()
