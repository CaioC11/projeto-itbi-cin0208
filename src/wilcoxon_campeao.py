import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from analise_f5 import PASTA_SAIDA, corrigir_holm, ler_folds, obter_baseline, pareamento_verificado
df = pd.read_csv(PASTA_SAIDA / "todas_configuracoes.csv")
validos = df[(df["conjunto"] == "completo") & (df["alvo_log"] == True)].sort_values("mae_media").reset_index(drop=True)
campeao = validos.iloc[0]
folds_camp = ler_folds(campeao["mae_por_fold"])
alvos = []
base = obter_baseline(df, "completo", True)
if base is not None: alvos.append(("Baseline (B0)", base))
sem_ream = validos[validos["desbalanceamento"].isin(["nenhum", "nao_aplicado"])]
if not sem_ream.empty: alvos.append(("Melhor G2 (sem reamostragem)", sem_ream.iloc[0]))
for pos in range(1, min(5, len(validos))):
    cand = validos.iloc[pos]
    alvos.append(("Top-" + str(pos + 1) + " Geral (" + str(cand["desbalanceamento"]) + ")", cand))
linhas, vistos = [], {campeao["chave_canonica"]}
for rotulo, ref in alvos:
    if ref["chave_canonica"] in vistos: continue
    vistos.add(ref["chave_canonica"])
    folds_ref = ler_folds(ref.get("mae_por_fold"))
    if folds_ref is None: continue
    dif = folds_camp - folds_ref
    stat, p_val = (0.0, 1.0) if np.all(dif == 0) else wilcoxon(dif, zero_method="wilcox", alternative="two-sided", method="auto")
    linhas.append({"comparacao": "Campeao G3 vs " + rotulo, "id_campeao": campeao.get("id"), "id_referencia": ref.get("id"), "mae_campeao": round(float(campeao["mae_media"]), 2), "mae_referencia": round(float(ref["mae_media"]), 2), "diferenca_mae": round(float(campeao["mae_media"] - ref["mae_media"]), 2), "vitorias_folds": str(int(np.sum(folds_camp < folds_ref))) + "/10", "pareado": pareamento_verificado(campeao, ref), "estatistica": float(stat), "p_valor": float(p_val)})
res = pd.DataFrame(linhas)
res["p_ajustado_holm"] = corrigir_holm(res["p_valor"].to_numpy())
res["significativo_5pct"] = res["pareado"] & (res["p_ajustado_holm"] < 0.05)
res.to_csv(PASTA_SAIDA / "wilcoxon_campeao_g3.csv", index=False)
print("=== WILCOXON PAREADO: CAMPEAO G3 VS REFERENCIAS ===")
print(res[["comparacao", "mae_campeao", "mae_referencia", "diferenca_mae", "vitorias_folds", "p_valor", "p_ajustado_holm", "significativo_5pct"]].to_string(index=False))
