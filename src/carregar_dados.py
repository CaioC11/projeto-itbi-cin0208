import pandas as pd

def carregar_dados(caminho="data/raw/raw_csv_itbi.csv"):
    df = pd.read_csv(caminho, sep=";", decimal=",", encoding="utf-8")
    for col in ["latitude", "longitude"]:
        df[col] = pd.to_numeric(df[col])
    return df