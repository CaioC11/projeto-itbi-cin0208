import pandas as pd
pd.set_option("display.width", 250)
b0 = pd.read_csv("resultados/referencias_f5_3.csv").iloc[0]
f4 = pd.read_csv("resultados/experimentos_f4.csv")
faixas = ["q1", "q2", "q3", "q4", "p99"]
linhas = [{"variante": "B0", **{f: b0[f + "_mae_media"] for f in faixas}, "mae_log": b0["mae_log_media"]}]
vies = [{"variante": "B0", **{f: b0[f + "_vies_media"] for f in faixas}}]
for _, r in f4.iterrows():
    linhas.append({"variante": r["desbalanceamento"], **{f: r[f + "_mae_media"] for f in faixas}, "mae_log": r["mae_log_media"]})
    vies.append({"variante": r["desbalanceamento"], **{f: r[f + "_vies_media"] for f in faixas}})
mae = pd.DataFrame(linhas).set_index("variante")
v = pd.DataFrame(vies).set_index("variante")
print("MAE por faixa (R$)"); print(mae.round(3).to_string())
print("\nVariacao do MAE vs B0 (%)"); print(((mae.drop(columns="mae_log") / mae.drop(columns="mae_log").loc["B0"] - 1) * 100).round(1).to_string())
print("\nViés por faixa (R$)"); print(v.round(0).to_string())
