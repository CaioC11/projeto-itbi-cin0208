"""F4-3: reroda as variantes de reamostragem no executor atual (com metricas por faixa)."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.executor import executar_grade
from src.reamostragem import montar_experimentos

SAIDA_F4 = RAIZ / "resultados" / "experimentos_f4.csv"

if __name__ == "__main__":
    experimentos = montar_experimentos(incluir_smogn=False)
    print(f"{len(experimentos)} experimentos -> {SAIDA_F4}")
    executar_grade(experimentos, salvar=True, saida=SAIDA_F4)
