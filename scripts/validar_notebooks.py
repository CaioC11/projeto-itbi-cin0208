"""Executa as análises com kernels novos e salva os resultados separadamente."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

RAIZ = Path(__file__).resolve().parents[1]


def main():
    destino = RAIZ / "resultados" / "notebooks"
    destino.mkdir(parents=True, exist_ok=True)
    for arquivo in sorted((RAIZ / "notebooks").glob("*.ipynb")):
        print(f"Executando {arquivo.name}...", flush=True)
        notebook = nbformat.read(arquivo, as_version=4)
        nbformat.validate(notebook)
        NotebookClient(
            notebook, timeout=300, kernel_name="python3",
            resources={"metadata": {"path": str(RAIZ / "notebooks")}},
        ).execute()
        nbformat.write(notebook, destino / arquivo.name)
        print(f"OK: {arquivo.name}", flush=True)


if __name__ == "__main__":
    main()
