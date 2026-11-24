import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = pd.read_csv("resultados/referencias_f5_3.csv")
b0 = d.iloc[0]
faixas = ["q1", "q2", "q3", "q4", "p99"]
mae = [b0[f + "_mae_media"] for f in faixas]
vies = [b0[f + "_vies_media"] for f in faixas]
rot = [f.upper() for f in faixas]

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(rot, mae)
ax[0].set_yscale("log")
ax[0].set_title("MAE por faixa do alvo (R$, escala log)")
ax[1].bar(rot, vies)
ax[1].axhline(0, color="black", linewidth=0.8)
ax[1].set_title("Viés por faixa (R$)")
plt.tight_layout()
plt.savefig("resultados/diagnostico_b0_faixas.png", dpi=150)
print("salvo")
