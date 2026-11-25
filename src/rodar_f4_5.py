"""F4-5 (Estagio 3): cruza o Top-5 do Gate G2 com as tecnicas de reamostragem."""
import argparse
import functools
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
SRC = RAIZ / "src"
for p in (str(RAIZ), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from executor import criar_chave, executar_experimento
from grade_f5 import chaves_existentes, criar_filtro_outliers, montar_pipeline
from reamostragem import random_over, random_under, ruido_gaussiano, smogn, smoter

TOP5_CSV = RAIZ / "resultados" / "top5_g2.csv"
SAIDA_F4_5 = RAIZ / "resultados" / "experimentos_f4_5.csv"

COLUNAS_CONFIG = [
    "faltantes",
    "outliers",
    "sfh",
    "encoding",
    "escala",
    "selecao",
    "pca",
    "ponderacao",
]


def obter_variantes(incluir_smogn=False):
    variantes = [
        ("random_over", random_over),
        ("random_under", random_under),
        ("gaussian_noise", ruido_gaussiano),
        ("smoter", smoter),
    ]
    if incluir_smogn:
        variantes.extend([
            ("smogn_0.5", functools.partial(smogn, rel_thres=0.5)),
            ("smogn_0.8", functools.partial(smogn, rel_thres=0.8)),
        ])
    return variantes


def montar_grade_f4_5(top5_csv=TOP5_CSV, incluir_smogn=False):
    top5 = pd.read_csv(top5_csv)
    variantes = obter_variantes(incluir_smogn=incluir_smogn)
    experimentos = []

    for rank, (_, row) in enumerate(top5.iterrows(), start=1):
        base_cfg = {col: row[col] for col in COLUNAS_CONFIG}
        for nome_var, func_var in variantes:
            cfg = dict(base_cfg)
            cfg["desbalanceamento"] = nome_var
            experimentos.append({
                "rank_g2": rank,
                "id_g2": row["chave"],
                "configuracao": cfg,
                "reamostrar": func_var,
            })

    return top5, experimentos


def imprimir_resumo(top5_df, saida_csv):
    if not Path(saida_csv).exists():
        return

    res = pd.read_csv(saida_csv)
    col_mae = "mae_media"
    col_r2 = "r2_media"
    col_p99 = "p99_mae_media"
    col_q4 = "q4_mae_media"

    print("\n=== RESUMO F4-5 (TOP-5 G2 x REAMOSTRAGEM) ===")
    cols_show = [
        "faltantes", "escala", "ponderacao", "desbalanceamento",
        col_mae, col_r2, col_q4, col_p99,
    ]
    cols_show = [c for c in cols_show if c in res.columns]
    ordenado = res.sort_values(col_mae)[cols_show]
    print(ordenado.to_string(index=False))

    melhor_g2 = top5_df.sort_values(col_mae).iloc[0]
    melhor_f45 = res.sort_values(col_mae).iloc[0]
    delta_mae = melhor_f45[col_mae] - melhor_g2[col_mae]
    delta_p99 = melhor_f45[col_p99] - melhor_g2[col_p99]

    print(
        f"\nMelhor G2 (sem reamostragem): MAE = R$ {melhor_g2[col_mae]:,.2f} | "
        f"R2 = {melhor_g2[col_r2]:.4f} | P99 MAE = R$ {melhor_g2[col_p99]:,.2f}"
    )
    print(
        f"Melhor F4-5 ({melhor_f45['desbalanceamento']}): MAE = R$ {melhor_f45[col_mae]:,.2f} "
        f"({delta_mae:+,.2f}) | R2 = {melhor_f45[col_r2]:.4f} | "
        f"P99 MAE = R$ {melhor_f45[col_p99]:,.2f} ({delta_p99:+,.2f})"
    )


def main():
    parser = argparse.ArgumentParser(description="F4-5: Top-5 G2 x Reamostragem")
    parser.add_argument("--incluir-smogn", action="store_true", help="Inclui smogn_0.5 e smogn_0.8")
    parser.add_argument("--saida", type=Path, default=SAIDA_F4_5)
    parser.add_argument("--sem-salvar", action="store_true")
    args = parser.parse_args()

    top5_df, experimentos = montar_grade_f4_5(incluir_smogn=args.incluir_smogn)
    salvar = not args.sem_salvar
    concluidas = chaves_existentes(args.saida) if salvar else set()

    print(f"F4-5: {len(experimentos)} experimentos -> {args.saida}", flush=True)

    for idx, exp in enumerate(experimentos, start=1):
        cfg = exp["configuracao"]
        chave = criar_chave(cfg, conjunto="completo", alvo_log=True)

        if chave in concluidas:
            print(f"[{idx}/{len(experimentos)}] Ja registrado: G2#{exp['rank_g2']} + {cfg['desbalanceamento']}")
            continue

        print(
            f"[{idx}/{len(experimentos)}] G2#{exp['rank_g2']} ({exp['id_g2'][:8]}) + "
            f"{cfg['desbalanceamento']} ({cfg['faltantes']} / {cfg['escala']} / {cfg['ponderacao']})",
            flush=True,
        )

        resultado = executar_experimento(
            configuracao=cfg,
            preprocessamento=montar_pipeline(cfg),
            filtro_treino=criar_filtro_outliers(cfg["outliers"]),
            reamostrar=exp["reamostrar"],
            conjunto="completo",
            alvo_log=True,
            salvar=salvar,
            saida=args.saida,
        )
        concluidas.add(chave)
        print(
            f"   -> MAE: R$ {resultado['mae_media']:,.2f} | "
            f"R2: {resultado['r2_media']:.4f} | "
            f"P99 MAE: R$ {resultado['p99_mae_media']:,.2f}",
            flush=True,
        )

    if salvar:
        imprimir_resumo(top5_df, args.saida)


if __name__ == "__main__":
    main()
