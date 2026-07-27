#!/usr/bin/env python3
"""Orquesta el pipeline COMPLETO de clasificación ML.

Encadena:
    1) Generación de recortes desde data/augmented/
    2) Entrenamiento de los 3 clasificadores
    3) Evaluación sobre TEST
    4) Tabla comparativa

Cada paso es tolerante a fallos: si un modelo revienta, el resto
continúa y al final se reporta qué funcionó y qué no.

Uso:
    python scripts/run_ml_pipeline.py                    # todo
    python scripts/run_ml_pipeline.py --skip_dataset     # asume crops ya generados
    python scripts/run_ml_pipeline.py --epochs 40        # entrena CNN por 40 epochs
    python scripts/run_ml_pipeline.py --overwrite_crops  # regenera recortes desde cero

Si querés control fino, usá los scripts individuales:
    scripts/run_ml_dataset.py
    scripts/run_ml_train.py    --model <name|all>
    scripts/run_ml_evaluate.py --model <name|all> --compare
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.classification.dataset_builder import build_crops, imprimir_resumen, load_split  # noqa: E402
from src.classification.evaluation import compare_reports, evaluate_model  # noqa: E402
from src.classification.models import EXTENSIONS, REGISTRY, get  # noqa: E402
from src.paths import CLASSIF_MODELS, DATA_AUG_DIR, ML_DATASETS_DIR, ML_REPORTS_DIR  # noqa: E402


def _paso(titulo: str) -> None:
    print("\n" + "#" * 68)
    print(f"# {titulo}")
    print("#" * 68)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip_dataset", action="store_true",
                    help="Salta la generación de recortes (asume que ya existen).")
    ap.add_argument("--overwrite_crops", action="store_true",
                    help="Borra los recortes anteriores y los regenera.")
    ap.add_argument("--epochs", type=int, default=25,
                    help="Epochs de la CNN.")
    args = ap.parse_args()

    crops_dir = ML_DATASETS_DIR / "crops"

    # ------------------------------------------------------------------ #
    _paso("PASO 1: Construir dataset de recortes")
    if args.skip_dataset:
        print("[INFO] --skip_dataset activo, se asume crops ya generados en", crops_dir)
    else:
        conteo = build_crops(dir_dataset=DATA_AUG_DIR, dir_salida=crops_dir,
                             sobrescribir=args.overwrite_crops)
        imprimir_resumen(conteo)

    # ------------------------------------------------------------------ #
    _paso("PASO 2: Cargar splits en memoria")
    X_tr, y_tr, class_names = load_split("train", dir_crops=crops_dir)
    X_va, y_va, _           = load_split("val",   dir_crops=crops_dir)
    X_te, y_te, _           = load_split("test",  dir_crops=crops_dir)
    print(f"train={len(X_tr)}  val={len(X_va)}  test={len(X_te)}  clases={len(class_names)}")

    # ------------------------------------------------------------------ #
    entrenados: dict[str, bool] = {}
    for name in REGISTRY:
        _paso(f"PASO 3.{name}: Entrenar {name}")
        try:
            mod = get(name)
            if name == "cnn":
                model = mod.train(X_tr, y_tr, class_names,
                                  X_val=X_va, y_val=y_va, epochs=args.epochs)
            else:
                model = mod.train(X_tr, y_tr, class_names)
            mod.save(model, CLASSIF_MODELS / f"{name}{EXTENSIONS[name]}")
            entrenados[name] = True
        except Exception:
            print(f"[ERROR] Falló {name}. Los demás modelos siguen.")
            traceback.print_exc()
            entrenados[name] = False

    # ------------------------------------------------------------------ #
    _paso("PASO 4: Evaluar en TEST")
    evaluados: dict[str, bool] = {}
    for name in REGISTRY:
        if not entrenados.get(name):
            print(f"[SKIP] {name} no se entrenó, se salta evaluación.")
            evaluados[name] = False
            continue
        try:
            mod = get(name)
            model = mod.load(CLASSIF_MODELS / f"{name}{EXTENSIONS[name]}")
            t0 = time.time()
            y_pred = mod.predict(model, X_te)
            t_pred = time.time() - t0
            evaluate_model(name, y_te, y_pred, class_names,
                           duracion_train_s=model.get("duracion_train_s") if isinstance(model, dict) else None,
                           duracion_pred_s=t_pred)
            evaluados[name] = True
        except Exception:
            print(f"[ERROR] Falló evaluación de {name}.")
            traceback.print_exc()
            evaluados[name] = False

    # ------------------------------------------------------------------ #
    _paso("PASO 5: Tabla comparativa")
    compare_reports(ML_REPORTS_DIR)

    _paso("RESUMEN FINAL")
    print(f"Reportes:   {ML_REPORTS_DIR}")
    print(f"Modelos:    {CLASSIF_MODELS}")
    print()
    for name in REGISTRY:
        e = "OK" if entrenados.get(name) else "FAIL"
        v = "OK" if evaluados.get(name) else "FAIL"
        print(f"  {name:<20} train:{e:<5}  eval:{v}")


if __name__ == "__main__":
    main()
