"""Construye un dataset de CLASIFICACIÓN a partir del dataset de detección.

Estrategia
----------
El dataset aumentado (``data/augmented/``) está en formato YOLO de detección:
cada imagen tiene varias bboxes con clases distintas. Para clasificación
necesitamos ``una etiqueta por muestra``, así que:

  1. Recorremos cada imagen del split (train/val/test).
  2. Por cada bbox de su .txt YOLO, recortamos la región y la guardamos
     como una imagen separada en ``outputs/ml_datasets/crops/<split>/<clase>/``.
  3. Cada recorte queda etiquetado por la carpeta que lo contiene
     (compatible con ``sklearn`` y ``torchvision.datasets.ImageFolder``).

Ventajas de guardar los recortes en disco:
  * Los tres clasificadores (HOG+SVM, ColorHist+RF, CNN) usan la MISMA
    fuente sin duplicar código.
  * Se pueden inspeccionar visualmente para verificar la calidad.
  * ImageFolder de PyTorch los consume directo.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from src.paths import DATA_AUG_DIR, ML_DATASETS_DIR


CROPS_ROOT: Path  = ML_DATASETS_DIR / "crops"
SPLITS = ("train", "val", "test")
MIN_CROP_PX = 24   # descartamos recortes casi vacíos (bboxes degenerados)


def _leer_classes(dir_dataset: Path) -> list[str]:
    ruta = dir_dataset / "classes.txt"
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}. ¿Corriste el augmentation_pipeline?")
    return [l.strip() for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]


def _leer_bboxes(ruta_txt: Path) -> list[tuple[int, float, float, float, float]]:
    """Devuelve lista de (class_id, cx, cy, w, h) desde un .txt YOLO."""
    if not ruta_txt.exists():
        return []
    salida = []
    for linea in ruta_txt.read_text(encoding="utf-8").splitlines():
        partes = linea.split()
        if len(partes) != 5:
            continue
        try:
            cid = int(float(partes[0]))
            cx, cy, w, h = map(float, partes[1:])
            salida.append((cid, cx, cy, w, h))
        except ValueError:
            continue
    return salida


def _recortar_bbox(img: np.ndarray, cx: float, cy: float, w: float, h: float,
                   margen: float = 0.05) -> np.ndarray | None:
    """Recorta la bbox YOLO normalizada con un margen relativo (default 5%)."""
    H, W = img.shape[:2]
    bw = w * (1 + 2 * margen)
    bh = h * (1 + 2 * margen)
    x1 = int(round((cx - bw / 2) * W))
    y1 = int(round((cy - bh / 2) * H))
    x2 = int(round((cx + bw / 2) * W))
    y2 = int(round((cy + bh / 2) * H))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < MIN_CROP_PX or y2 - y1 < MIN_CROP_PX:
        return None
    return img[y1:y2, x1:x2].copy()


def build_crops(dir_dataset: Path = DATA_AUG_DIR,
                dir_salida: Path = CROPS_ROOT,
                sobrescribir: bool = False) -> dict[str, dict[str, int]]:
    """
    Genera los recortes en disco y devuelve un conteo {split: {clase: n}}.

    Si ``sobrescribir=False`` y ya existen recortes, se mantienen (idempotente).
    """
    dir_dataset = Path(dir_dataset)
    dir_salida = Path(dir_salida)
    clases = _leer_classes(dir_dataset)

    if sobrescribir and dir_salida.exists():
        import shutil
        shutil.rmtree(dir_salida)
    dir_salida.mkdir(parents=True, exist_ok=True)

    conteo: dict[str, dict[str, int]] = {s: {c: 0 for c in clases} for s in SPLITS}

    for split in SPLITS:
        img_dir = dir_dataset / "images" / split
        lbl_dir = dir_dataset / "labels" / split
        if not img_dir.is_dir():
            print(f"[WARN] Split {split} no encontrado ({img_dir}), se salta.")
            continue

        # Crear subcarpetas por clase
        for clase in clases:
            (dir_salida / split / clase).mkdir(parents=True, exist_ok=True)

        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            bboxes = _leer_bboxes(lbl_dir / (img_path.stem + ".txt"))
            for i, (cid, cx, cy, w, h) in enumerate(bboxes):
                if cid < 0 or cid >= len(clases):
                    continue
                crop = _recortar_bbox(img, cx, cy, w, h)
                if crop is None:
                    continue
                clase = clases[cid]
                destino = dir_salida / split / clase / f"{img_path.stem}_{i:02d}.jpg"
                if destino.exists() and not sobrescribir:
                    conteo[split][clase] += 1
                    continue
                cv2.imwrite(str(destino), crop)
                conteo[split][clase] += 1

    return conteo


def load_split(split: str,
               dir_crops: Path = CROPS_ROOT) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    """
    Carga en memoria un split como (X_bgr, y, class_names).

    X_bgr: lista de arrays BGR uint8 de tamaño variable.
    y:     array int64 con los class_id.
    """
    dir_crops = Path(dir_crops) / split
    if not dir_crops.is_dir():
        raise FileNotFoundError(f"No hay recortes en {dir_crops}. Corre build_crops() primero.")

    class_names = sorted([p.name for p in dir_crops.iterdir() if p.is_dir()])
    X: list[np.ndarray] = []
    y: list[int] = []
    for cid, clase in enumerate(class_names):
        for img_path in sorted((dir_crops / clase).iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            X.append(img)
            y.append(cid)
    return X, np.asarray(y, dtype=np.int64), class_names


def imprimir_resumen(conteo: dict[str, dict[str, int]]) -> None:
    """Imprime una tabla legible del conteo por split y clase."""
    if not conteo:
        print("(sin datos)")
        return
    clases = list(next(iter(conteo.values())).keys())
    print("\n" + "=" * 66)
    print("RECORTES POR SPLIT / CLASE")
    print("=" * 66)
    encab = f"{'clase':<20}" + "".join(f"{s:>10}" for s in SPLITS) + f"{'TOTAL':>10}"
    print(encab)
    print("-" * len(encab))
    for c in clases:
        fila = f"{c:<20}"
        total = 0
        for s in SPLITS:
            n = conteo.get(s, {}).get(c, 0)
            fila += f"{n:>10}"
            total += n
        fila += f"{total:>10}"
        print(fila)
    print("-" * len(encab))
    totales = f"{'TOTAL':<20}"
    gran_total = 0
    for s in SPLITS:
        n = sum(conteo.get(s, {}).values())
        totales += f"{n:>10}"
        gran_total += n
    totales += f"{gran_total:>10}"
    print(totales)
