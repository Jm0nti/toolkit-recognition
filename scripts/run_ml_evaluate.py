#!/usr/bin/env python3
"""Evalúa uno o varios clasificadores sobre el split TEST.

Uso:
    python scripts/run_ml_evaluate.py --model hog_svm
    python scripts/run_ml_evaluate.py --model all
    python scripts/run_ml_evaluate.py --model all --compare

Genera para cada modelo (en outputs/ml_reports/):
    <model>_metrics.json           métricas globales + por clase
    <model>_confusion_matrix.png   matriz normalizada por fila

Con --compare, además genera:
    comparison.md / comparison.csv / comparison.png
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.classification.dataset_builder import load_split  # noqa: E402
from src.classification.evaluation import compare_reports, evaluate_model  # noqa: E402
from src.classification.models import EXTENSIONS, REGISTRY, get  # noqa: E402
from src.paths import CLASSIF_MODELS, ML_DATASETS_DIR, ML_REPORTS_DIR  # noqa: E402


def _evaluate_one(name: str, X_te, y_te, class_names) -> bool:
    print("\n" + "=" * 66)
    print(f"EVALUANDO: {name}")
    print("=" * 66)
    mod = get(name)
    ruta_pesos = CLASSIF_MODELS / f"{name}{EXTENSIONS[name]}"
    if not ruta_pesos.exists():
        print(f"[SKIP] No existen pesos para {name} ({ruta_pesos}). "
              f"Corre run_ml_train.py --model {name} primero.")
        return False
    try:
        model = mod.load(ruta_pesos)
        t0 = time.time()
        y_pred = mod.predict(model, X_te)
        t_pred = time.time() - t0
        evaluate_model(
            name, y_te, y_pred, class_names,
            duracion_train_s=model.get("duracion_train_s") if isinstance(model, dict) else None,
            duracion_pred_s=t_pred,
        )
        return True
    except Exception:
        print(f"[ERROR] Falló evaluación de {name}:")
        traceback.print_exc()
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="all",
                    choices=[*REGISTRY.keys(), "all"])
    ap.add_argument("--crops_dir", default=str(ML_DATASETS_DIR / "crops"))
    ap.add_argument("--compare", action="store_true",
                    help="Genera la tabla comparativa final.")
    args = ap.parse_args()

    modelos = list(REGISTRY.keys()) if args.model == "all" else [args.model]

    print(f"[INFO] Cargando split test desde {args.crops_dir}...")
    X_te, y_te, class_names = load_split("test", dir_crops=Path(args.crops_dir))
    print(f"[INFO] test={len(X_te)}  clases={len(class_names)}")
    print(f"[INFO] Reportes -> {ML_REPORTS_DIR}")

    resultados: dict[str, bool] = {}
    for m in modelos:
        resultados[m] = _evaluate_one(m, X_te, y_te, class_names)

    print("\n" + "=" * 66)
    print("RESUMEN DE EVALUACIÓN")
    print("=" * 66)
    for m, ok in resultados.items():
        estado = "OK  " if ok else "SKIP/FAIL"
        print(f"  [{estado}] {m}")

    if args.compare:
        print()
        compare_reports(ML_REPORTS_DIR)


if __name__ == "__main__":
    main()
