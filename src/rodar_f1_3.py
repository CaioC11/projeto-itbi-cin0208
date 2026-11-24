"""Executa e registra os 4 experimentos de F1-3 (outliers) em resultados/experimentos_f1_3.csv."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.executor import executar_experimento
from src.encoding import ORDEM_ACABAMENTO, OrdinalOrdenado, OneHotRaras, montar_pipeline
from src.outliers import FiltroIQRLog, Winsorizar

SAIDA_F1_3 = RAIZ / "resultados" / "experimentos_f1_3.csv"

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

PASSOS_B0 = [
    ("ord", OrdinalOrdenado(ORDEM_ACABAMENTO)),
    ("oh", OneHotRaras(["bairro", "tipo_imovel", "tipo_construcao", "tipo_ocupacao", "estado_conservacao"])),
]


def rodar(nome, passos, filtro=None):
    cfg = {**CONFIG_B0, "outliers": nome}
    pipe = montar_pipeline(passos, estrategia_num="median")
    res = executar_experimento(
        configuracao=cfg,
        preprocessamento=pipe,
        filtro_treino=filtro,
        salvar=True,
        saida=SAIDA_F1_3,
    )
    print(f"[{nome}] MAE={res['mae_media']:.2f} | R2={res['r2_media']:.4f} | cols={res['n_colunas']}")


if __name__ == "__main__":
    rodar("iqr_1.5", PASSOS_B0, FiltroIQRLog(1.5))
    rodar("iqr_3.0", PASSOS_B0, FiltroIQRLog(3.0))
    rodar("winsor_1_99", [("wins", Winsorizar(1, 99))] + PASSOS_B0)
    rodar("winsor_5_95", [("wins", Winsorizar(5, 95))] + PASSOS_B0)
