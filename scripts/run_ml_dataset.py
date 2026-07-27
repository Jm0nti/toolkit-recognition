#!/usr/bin/env python3
"""Genera el dataset de recortes para clasificación desde data/augmented/.

Uso:
    python scripts/run_ml_dataset.py                 # incremental
    python scripts/run_ml_dataset.py --overwrite     # regenera todo

Los recortes quedan en outputs/ml_datasets/crops/{train,val,test}/<clase>/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.classification.dataset_builder import build_crops, imprimir_resumen  # noqa: E402
from src.paths import DATA_AUG_DIR, ML_DATASETS_DIR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--augmented_dir", default=str(DATA_AUG_DIR),
                    help="Ruta del dataset aumentado con labels YOLO.")
    ap.add_argument("--out_dir", default=str(ML_DATASETS_DIR / "crops"),
                    help="Ruta donde se guardan los recortes.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Borra los recortes anteriores antes de generar.")
    args = ap.parse_args()

    print(f"[INFO] Generando recortes desde {args.augmented_dir}")
    print(f"[INFO] Destino: {args.out_dir}  overwrite={args.overwrite}")
    conteo = build_crops(dir_dataset=Path(args.augmented_dir),
                         dir_salida=Path(args.out_dir),
                         sobrescribir=args.overwrite)
    imprimir_resumen(conteo)


if __name__ == "__main__":
    main()
