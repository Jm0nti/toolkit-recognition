#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Entrena el segundo modelo ("real6") con el 6-tool dataset.

Requiere haber corrido antes scripts/run_dataset_masivo.py (que descarga
las imágenes y crea data/masivo/data.yaml). Deja los pesos en
models/detection/real6/best.pt, donde la página los ofrece como
"6-tool · dataset real" en el selector de modelo.

OJO: el dataset completo son ~20k imágenes; en un portátil sin GPU eso
son días. Recomendado: entrenar con una muestra
(run_dataset_masivo.py --limit 1500) o en un PC con GPU / Colab.

Uso:
    python scripts/run_train_masivo.py --epochs 30
    python scripts/run_train_masivo.py --model_size s --epochs 50
"""
import argparse
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.detection.detector import train  # noqa: E402
from src.paths import DATA_DIR, DETECTION_MODELS, RUNS_DIR  # noqa: E402

DATA_YAML = DATA_DIR / "masivo" / "data.yaml"
DESTINO = DETECTION_MODELS / "real6" / "best.pt"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--model_size", type=str, default="n",
                    help="n (nano, rápido) o s (small, más preciso)")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="tamaño de imagen; 480 entrena casi el doble de rápido")
    args = ap.parse_args()

    if not DATA_YAML.exists():
        raise SystemExit("No existe data/masivo/data.yaml. "
                         "Corre primero: python scripts/run_dataset_masivo.py")

    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    train(dataset_yaml=str(DATA_YAML),
          model_size=args.model_size,
          epochs=args.epochs,
          imgsz=args.imgsz,
          name="real6",
          output_weights=str(DESTINO))

    # por si train() no copió los pesos, los copio del run
    origen = RUNS_DIR / "real6" / "weights" / "best.pt"
    if not DESTINO.exists() and origen.exists():
        shutil.copy2(origen, DESTINO)
    print(f"\nModelo real6 listo en {DESTINO}")
    print("Reinicia la página y aparece en el selector como '6-tool · dataset real'.")


if __name__ == "__main__":
    main()
