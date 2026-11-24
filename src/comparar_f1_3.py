import pandas as pd
pd.set_option("display.width", 250)
b0 = pd.read_csv("resultados/referencias_f5_3.csv").iloc[0]
f = pd.read_csv("resultados/experimentos_f1_3.csv")
faixas = ["q1", "q2", "q3", "q4", "p99"]
linhas = [{"variante": "B0", **{k: b0[k + "_mae_media"] for k in faixas}, "mae_log": b0["mae_log_media"]}]
for _, r in f.iterrows():
    linhas.append({"variante": r["outliers"], **{k: r[k + "_mae_media"] for k in faixas}, "mae_log": r["mae_log_media"]})
m = pd.DataFrame(linhas).set_index("variante")
print("MAE por faixa (R$)"); print(m.round(3).to_string())
print("\nVariacao vs B0 (%)"); print(((m / m.loc["B0"] - 1) * 100).round(1).to_string())
