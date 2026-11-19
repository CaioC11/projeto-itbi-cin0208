"""Executa e registra os experimentos oficiais de F1-2, F1-4, F2-1 e F2-2 no CSV."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.executor import executar_experimento
from src.limpeza import TratamentoFaltantes, TratamentoSFH
from src.encoding import (
    ORDEM_ACABAMENTO,
    ORDENS_ORDINAL,
    COLUNAS_ALTA_CARDINALIDADE,
    COLUNAS_BAIXA_CARDINALIDADE,
    COLUNAS_NOMINAIS_ORDINAL,
    OrdinalOrdenado,
    OneHotRaras,
    FrequencyEncoding,
    TargetEncodingSuave,
    TargetEncodingForte,
    OneHotPesoReduzido,
    DropColunas,
    montar_pipeline,
)

CONFIG_B0 = {
    "faltantes": "mediana_global",
    "outliers": "nenhum",
    "sfh": "original",
    "encoding": "onehot",
    "escala": "standard",
    "selecao": "nenhuma",
    "pca": "nenhum",
    "ponderacao": "nenhuma",
    "desbalanceamento": "nenhum",
}

PIPE_B0 = [
    ("ord", OrdinalOrdenado(ORDEM_ACABAMENTO)),
    ("oh", OneHotRaras(["bairro", "tipo_imovel", "tipo_construcao", "tipo_ocupacao", "estado_conservacao"])),
]


def rodar(nome, override, passos, conjunto="completo", estrategia_num="median"):
    cfg = {**CONFIG_B0, **override}
    pipe = montar_pipeline(passos, estrategia_num=estrategia_num)
    res = executar_experimento(
        configuracao=cfg,
        preprocessamento=pipe,
        conjunto=conjunto,
        salvar=True,
    )
    print(f"[{nome}] MAE={res['mae_media']:.2f} | R2={res['r2_media']:.4f} | cols={res['n_colunas']}")


def main():
    print("=== F2-1: ENCODING (4 variantes) ===")
    rodar("F2-1 Ordinal", {"encoding": "ordinal"},
          [("ord", OrdinalOrdenado(ORDENS_ORDINAL)), ("oh", OneHotRaras(COLUNAS_NOMINAIS_ORDINAL))])
    rodar("F2-1 Frequency", {"encoding": "frequency"},
          [("ord", OrdinalOrdenado(ORDEM_ACABAMENTO)), ("freq", FrequencyEncoding(COLUNAS_ALTA_CARDINALIDADE)), ("oh", OneHotRaras(COLUNAS_BAIXA_CARDINALIDADE))])
    rodar("F2-1 Target Suave (10)", {"encoding": "target_suave_10"},
          [("ord", OrdinalOrdenado(ORDEM_ACABAMENTO)), ("te", TargetEncodingSuave(COLUNAS_ALTA_CARDINALIDADE, 10)), ("oh", OneHotRaras(COLUNAS_BAIXA_CARDINALIDADE))])
    rodar("F2-1 Target Forte (100)", {"encoding": "target_forte_100"},
          [("ord", OrdinalOrdenado(ORDEM_ACABAMENTO)), ("te", TargetEncodingForte(COLUNAS_ALTA_CARDINALIDADE, 100)), ("oh", OneHotRaras(COLUNAS_BAIXA_CARDINALIDADE))])

    print("\n=== F2-2: VARIANTES EXTRAS (4 variantes) ===")
    cols_resto = ["tipo_imovel", "tipo_construcao", "tipo_ocupacao", "estado_conservacao"]
    rodar("F2-2 Híbrido", {"encoding": "hibrido_te_bairro_oh_resto"},
          [("ord", OrdinalOrdenado(ORDEM_ACABAMENTO)), ("te_b", TargetEncodingSuave(["bairro"], 10)), ("oh", OneHotRaras(cols_resto))])
    rodar("F2-2 Bairro Raros (<30)", {"encoding": "onehot_bairro_raros_30"},
          [("ord", OrdinalOrdenado(ORDEM_ACABAMENTO)), ("oh_b", OneHotRaras(["bairro"], min_count=30)), ("oh", OneHotRaras(cols_resto))])
    rodar("F2-2 Sem Bairro", {"encoding": "onehot_sem_bairro"},
          [("drop_b", DropColunas(["bairro"])), ("ord", OrdinalOrdenado(ORDEM_ACABAMENTO)), ("oh", OneHotRaras(cols_resto))])
    rodar("F2-2 OneHot Peso 0.5", {"encoding": "onehot_peso_0.5"},
          [("ord", OrdinalOrdenado(ORDEM_ACABAMENTO)), ("oh_p", OneHotPesoReduzido(["bairro"] + cols_resto, peso=0.5))])

    print("\n=== F1-4: VARIANTES SFH (2 variantes) ===")
    rodar("F1-4 Sem SFH", {"sfh": "remover"}, [("sfh", TratamentoSFH("remover"))] + PIPE_B0)
    rodar("F1-4 SFH Binarizado", {"sfh": "binarizar"}, [("sfh", TratamentoSFH("binarizar"))] + PIPE_B0)

    print("\n=== F1-2: FALTANTES (6 variantes) ===")
    rodar("F1-2 remover_linhas", {"faltantes": "remover_linhas"}, PIPE_B0, conjunto="sem_faltantes")
    for est in ["mediana_bairro", "mediana_bairro_indicador", "sem_lat_lon", "area_terreno_tipo", "media"]:
        est_num = "mean" if est == "media" else "median"
        rodar(f"F1-2 {est}", {"faltantes": est}, [("imp", TratamentoFaltantes(est))] + PIPE_B0, estrategia_num=est_num)


if __name__ == "__main__":
    main()