import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from analise_f5 import PASTA_SAIDA, TECNICAS, obter_baseline
from baseline import ALVO, CATEGORICAS, NUMERICAS, carregar_e_limpar
from grade_f5 import criar_filtro_outliers, montar_pipeline
from reamostragem import random_over, random_under, ruido_gaussiano, smoter
MAPA_REAM = {"nenhum": None, "nao_aplicado": None, "random_over": random_over, "random_under": random_under, "gaussian_noise": ruido_gaussiano, "smoter": smoter}
df_res = pd.read_csv(PASTA_SAIDA / "todas_configuracoes.csv")
validos = df_res[(df_res["conjunto"] == "completo") & (df_res["alvo_log"] == True)].sort_values("mae_media")
campeao = validos.iloc[0]
cfg = {c: str(campeao[c]) for c in TECNICAS}
filtro_fn = criar_filtro_outliers(cfg["outliers"])
ream_fn = MAPA_REAM[cfg["desbalanceamento"]]
df = carregar_e_limpar()
X, y = df[NUMERICAS + CATEGORICAS], df[ALVO].to_numpy()
cv = KFold(n_splits=10, shuffle=True, random_state=42)
obs, pred, maes_fold = np.empty(len(y)), np.empty(len(y)), []
for tr, te in cv.split(X):
    X_tr, y_tr = X.iloc[tr].copy(), y[tr].copy()
    if filtro_fn is not None: X_tr, y_tr = filtro_fn(X_tr, y_tr)
    y_aj = np.log1p(np.asarray(y_tr).ravel())
    prep = clone(montar_pipeline(cfg))
    X_tr_m = prep.fit_transform(X_tr, y_aj)
    X_te_m = prep.transform(X.iloc[te])
    if ream_fn is not None: X_tr_m, y_aj = ream_fn(X_tr_m, y_aj)
    knn = KNeighborsRegressor(n_neighbors=5, weights="uniform", metric="minkowski", p=2).fit(X_tr_m, y_aj)
    obs[te], pred[te] = y[te], np.expm1(knn.predict(X_te_m))
    maes_fold.append(mean_absolute_error(obs[te], pred[te]))
residuos = obs - pred
pd.DataFrame({"observado": obs, "previsto": pred, "residuo": residuos}).to_csv(PASTA_SAIDA / "predicoes_campeao_g3.csv", index=False)
print(f"Campeao G3 ({cfg['desbalanceamento']}): MAE medio folds = R$ {np.mean(maes_fold):,.2f} | R2 OOF = {r2_score(obs, pred):.4f}")
plt.figure(figsize=(8, 5))
plt.scatter(obs, pred, s=6, alpha=0.3, color="#1f77b4")
lim = max(np.max(obs), np.max(pred))
plt.plot([0, lim], [0, lim], linestyle="--", color="black")
plt.xlabel("Valor observado (R$)")
plt.ylabel("Valor previsto (R$)")
plt.title("Campeao G3 (random_over): previsto x observado")
plt.tight_layout()
plt.savefig(PASTA_SAIDA / "previsto_observado_campeao_g3.png", dpi=150)
plt.close()
plt.figure(figsize=(8, 5))
plt.scatter(pred, residuos, s=6, alpha=0.3, color="#1f77b4")
plt.axhline(0, linestyle="--", color="black")
plt.xlabel("Valor previsto (R$)")
plt.ylabel("Residuo: observado - previsto (R$)")
plt.title("Campeao G3 (random_over): grafico de residuos")
plt.tight_layout()
plt.savefig(PASTA_SAIDA / "residuos_campeao_g3.png", dpi=150)
plt.close()
base = obter_baseline(df_res, "completo", True)
g2_top = validos[validos["desbalanceamento"].isin(["nenhum", "nao_aplicado"])].iloc[0]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
etapas = ["Baseline (B0)", "Melhor G2", "Campeao G3"]
maes = [base["mae_media"] / 1000.0, g2_top["mae_media"] / 1000.0, campeao["mae_media"] / 1000.0]
barras = axes[0].bar(etapas, maes, color=["#7f8c8d", "#2980b9", "#27ae60"])
axes[0].set_ylabel("MAE medio (R$ mil)")
axes[0].set_title("Evolucao do MAE Global por Estagio")
axes[0].set_ylim(0, max(maes) * 1.18)
for b, v in zip(barras, maes): axes[0].text(b.get_x() + b.get_width() / 2, v + 2.5, f"R$ {v:.1f}k", ha="center", fontsize=9, fontweight="bold")
faixas, x, w = ["Q1", "Q2", "Q3", "Q4"], np.arange(4), 0.26
axes[1].bar(x - w, [base[f.lower() + "_mae_media"] / 1000.0 for f in faixas], width=w, label="Baseline (B0)", color="#7f8c8d")
axes[1].bar(x, [g2_top[f.lower() + "_mae_media"] / 1000.0 for f in faixas], width=w, label="Melhor G2", color="#2980b9")
axes[1].bar(x + w, [campeao[f.lower() + "_mae_media"] / 1000.0 for f in faixas], width=w, label="Campeao G3", color="#27ae60")
axes[1].set_xticks(x)
axes[1].set_xticklabels(faixas)
axes[1].set_ylabel("MAE por quartil (R$ mil)")
axes[1].set_title("Reducao do Erro por Faixa de Valor (Q1 a Q4)")
axes[1].legend()
plt.tight_layout()
plt.savefig(PASTA_SAIDA / "comparativo_b0_vs_campeao.png", dpi=150)
plt.close()
print("Graficos salvos em resultados/analise_f5/!")
