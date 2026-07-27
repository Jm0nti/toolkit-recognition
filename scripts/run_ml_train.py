#!/usr/bin/env python3
"""Entrena uno o varios clasificadores de recortes.

Uso:
    python scripts/run_ml_train.py --model hog_svm
    python scripts/run_ml_train.py --model color_hist_rf
    python scripts/run_ml_train.py --model cnn --epochs 30
    python scripts/run_ml_train.py --model all

Si un modelo falla, los demás siguen: cada uno se entrena en un try/except
independiente. Al final se imprime un resumen de éxitos/fallos.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.classification.dataset_builder import load_split  # noqa: E402
from src.classification.models import EXTENSIONS, REGISTRY, get  # noqa: E402
from src.paths import CLASSIF_MODELS, ML_DATASETS_DIR  # noqa: E402


def _train_one(name: str, X_tr, y_tr, class_names, X_va, y_va, epochs: int) -> bool:
    print("\n" + "=" * 66)
    print(f"ENTRENANDO: {name}")
    print("=" * 66)
    mod = get(name)
    try:
        # Solo la CNN acepta datos de validación; los sklearn los ignoran.
        if name == "cnn":
            model = mod.train(X_tr, y_tr, class_names,
                              X_val=X_va, y_val=y_va, epochs=epochs)
        else:
            model = mod.train(X_tr, y_tr, class_names)
        destino = CLASSIF_MODELS / f"{name}{EXTENSIONS[name]}"
        mod.save(model, destino)
        return True
    except Exception:
        print(f"[ERROR] Falló entrenamiento de {name}:")
        traceback.print_exc()
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="all",
                    choices=[*REGISTRY.keys(), "all"],
                    help="Modelo a entrenar (o 'all').")
    ap.add_argument("--epochs", type=int, default=25,
                    help="Epochs de la CNN (los sklearn lo ignoran).")
    ap.add_argument("--crops_dir", default=str(ML_DATASETS_DIR / "crops"))
    args = ap.parse_args()

    modelos = list(REGISTRY.keys()) if args.model == "all" else [args.model]

    print(f"[INFO] Cargando splits desde {args.crops_dir}...")
    X_tr, y_tr, class_names = load_split("train", dir_crops=Path(args.crops_dir))
    X_va, y_va, _           = load_split("val",   dir_crops=Path(args.crops_dir))
    print(f"[INFO] train={len(X_tr)}  val={len(X_va)}  clases={len(class_names)}")

    resultados: dict[str, bool] = {}
    for m in modelos:
        resultados[m] = _train_one(m, X_tr, y_tr, class_names, X_va, y_va, args.epochs)

    print("\n" + "=" * 66)
    print("RESUMEN DE ENTRENAMIENTO")
    print("=" * 66)
    for m, ok in resultados.items():
        estado = "OK " if ok else "FAIL"
        print(f"  [{estado}] {m}")

    if not all(resultados.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
