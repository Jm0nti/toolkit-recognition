"""Métricas y reportes comparativos para los clasificadores.

Genera para cada modelo:
  * JSON con métricas globales (accuracy, macro-F1) y por clase (P/R/F1).
  * PNG con la matriz de confusión.

Y un helper ``compare_reports`` que junta todos los JSON en:
  * Una tabla comparativa en Markdown y CSV.
  * Un gráfico de barras comparando accuracy y macro-F1.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin pantalla para poder correr en servidor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.paths import ML_REPORTS_DIR


def evaluate_model(model_name: str,
                   y_true: np.ndarray,
                   y_pred: np.ndarray,
                   class_names: list[str],
                   dir_salida: Path = ML_REPORTS_DIR,
                   duracion_train_s: float | None = None,
                   duracion_pred_s: float | None = None) -> dict:
    """
    Calcula métricas, guarda JSON + matriz de confusión y devuelve el dict.
    """
    dir_salida = Path(dir_salida)
    dir_salida.mkdir(parents=True, exist_ok=True)

    acc      = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    reporte  = classification_report(
        y_true, y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    resumen = {
        "model": model_name,
        "accuracy": acc,
        "macro_f1": f1_macro,
        "duracion_train_s": duracion_train_s,
        "duracion_pred_s": duracion_pred_s,
        "n_test": int(len(y_true)),
        "per_class": {
            c: {
                "precision": reporte[c]["precision"],
                "recall":    reporte[c]["recall"],
                "f1":        reporte[c]["f1-score"],
                "support":   reporte[c]["support"],
            } for c in class_names
        },
    }

    ruta_json = dir_salida / f"{model_name}_metrics.json"
    ruta_json.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

    # Matriz de confusión (normalizada por fila = recall visual por clase)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plot_confusion_matrix(cm, class_names,
                          dir_salida / f"{model_name}_confusion_matrix.png",
                          titulo=f"Matriz de confusión — {model_name}")

    print(f"[{model_name}] acc={acc:.3f}  macro-F1={f1_macro:.3f}  n_test={len(y_true)}")
    return resumen


def plot_confusion_matrix(cm: np.ndarray,
                          class_names: list[str],
                          ruta_salida: Path,
                          titulo: str = "Matriz de confusión") -> None:
    """Guarda una matriz de confusión normalizada por fila."""
    cm_norm = cm.astype(np.float64) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           title=titulo, ylabel="Etiqueta real", xlabel="Predicción")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            valor = cm_norm[i, j]
            color = "white" if valor > 0.5 else "black"
            ax.text(j, i, f"{valor:.2f}", ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=140)
    plt.close(fig)


def compare_reports(dir_reports: Path = ML_REPORTS_DIR) -> dict:
    """
    Junta los ``*_metrics.json`` de dir_reports en una tabla comparativa.
    Guarda ``comparison.md``, ``comparison.csv`` y ``comparison.png``.
    """
    dir_reports = Path(dir_reports)
    json_files = sorted(dir_reports.glob("*_metrics.json"))
    if not json_files:
        print(f"[WARN] No hay reportes en {dir_reports}. Entrena algún modelo primero.")
        return {}

    filas = []
    for jf in json_files:
        data = json.loads(jf.read_text(encoding="utf-8"))
        filas.append({
            "model":            data["model"],
            "accuracy":         data["accuracy"],
            "macro_f1":         data["macro_f1"],
            "n_test":           data["n_test"],
            "duracion_train_s": data.get("duracion_train_s"),
            "duracion_pred_s":  data.get("duracion_pred_s"),
        })

    # Ordenar por accuracy descendente
    filas.sort(key=lambda r: r["accuracy"], reverse=True)

    # CSV
    csv_path = dir_reports / "comparison.csv"
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write("model,accuracy,macro_f1,n_test,duracion_train_s,duracion_pred_s\n")
        for r in filas:
            dt = "" if r["duracion_train_s"] is None else f"{r['duracion_train_s']:.2f}"
            dp = "" if r["duracion_pred_s"]  is None else f"{r['duracion_pred_s']:.3f}"
            fh.write(f"{r['model']},{r['accuracy']:.4f},{r['macro_f1']:.4f},"
                     f"{r['n_test']},{dt},{dp}\n")

    # Markdown
    md_path = dir_reports / "comparison.md"
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("# Comparación de clasificadores\n\n")
        fh.write("| Modelo | Accuracy | Macro-F1 | N test | Train (s) | Pred (s) |\n")
        fh.write("|---|---:|---:|---:|---:|---:|\n")
        for r in filas:
            dt = "—" if r["duracion_train_s"] is None else f"{r['duracion_train_s']:.2f}"
            dp = "—" if r["duracion_pred_s"]  is None else f"{r['duracion_pred_s']:.3f}"
            fh.write(f"| {r['model']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | "
                     f"{r['n_test']} | {dt} | {dp} |\n")

    # Gráfico
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(filas)), 4.5))
    x = np.arange(len(filas))
    ancho = 0.35
    ax.bar(x - ancho / 2, [r["accuracy"] for r in filas], ancho, label="Accuracy")
    ax.bar(x + ancho / 2, [r["macro_f1"] for r in filas], ancho, label="Macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels([r["model"] for r in filas], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Comparación de clasificadores (split test)")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(dir_reports / "comparison.png", dpi=140)
    plt.close(fig)

    print(f"[OK] Comparación guardada en:")
    print(f"     - {md_path}")
    print(f"     - {csv_path}")
    print(f"     - {dir_reports / 'comparison.png'}")
    return {"rows": filas, "md": str(md_path), "csv": str(csv_path)}
