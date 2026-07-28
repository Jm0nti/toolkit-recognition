#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Demo visual de la extracción de características.

Genera dos evidencias para la página y el informe:
  * hog.jpg        — recorte en gris y su visualización HOG (gradientes/silueta).
  * histograma.png — histograma de color por canal HSV del mismo recorte.

Uso:
    python scripts/run_features_demo.py
    python scripts/run_features_demo.py --img data/raw/tiago/6S.jpg
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.classification.preprocessing import resize_pad, to_grayscale  # noqa: E402
from src.paths import DATA_RAW_DIR, OUTPUTS_DIR  # noqa: E402

SALIDA_DIR = OUTPUTS_DIR / "caracteristicas"


def _primera_foto() -> Path:
    for carpeta in sorted(DATA_RAW_DIR.iterdir()):
        if carpeta.is_dir():
            for p in sorted(carpeta.iterdir()):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    return p
    raise SystemExit("No encontré fotos en data/raw/")


def rotular(img, texto):
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (25, 25, 25), -1)
    cv2.putText(img, texto, (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def demo_hog(img_bgr: np.ndarray) -> np.ndarray:
    """Recorte en gris + visualización HOG, lado a lado."""
    canvas = resize_pad(img_bgr, 256)
    gray = to_grayscale(canvas)
    _, hog_img = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                     cells_per_block=(2, 2), block_norm="L2-Hys",
                     transform_sqrt=True, visualize=True)
    hog_img = exposure.rescale_intensity(hog_img, in_range=(0, hog_img.max() or 1))
    hog_u8 = (hog_img * 255).astype(np.uint8)
    gris3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    hog3 = cv2.cvtColor(hog_u8, cv2.COLOR_GRAY2BGR)
    return cv2.hconcat([rotular(gris3, "gris"), rotular(hog3, "HOG (forma)")])


def demo_histograma(img_bgr: np.ndarray, destino: Path) -> None:
    """Histograma de color por canal HSV."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    canales = [("H (matiz)", 0, 180, "#f2b807"),
               ("S (saturación)", 1, 256, "#d1493f"),
               ("V (valor)", 2, 256, "#5aa9d6")]
    plt.figure(figsize=(7, 3.2))
    for nombre, ch, rango, color in canales:
        hist = cv2.calcHist([hsv], [ch], None, [32], [0, rango]).flatten()
        hist = hist / (hist.sum() + 1e-9)
        plt.plot(np.linspace(0, 1, 32), hist, color=color, label=nombre, linewidth=2)
    plt.title("Histograma de color HSV")
    plt.xlabel("valor del canal (normalizado)")
    plt.ylabel("frecuencia")
    plt.legend()
    plt.tight_layout()
    plt.savefig(destino, dpi=110)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--img", type=str, default="", help="ruta de la foto a usar")
    args = ap.parse_args()

    ruta = Path(args.img) if args.img else _primera_foto()
    img = cv2.imread(str(ruta))
    if img is None:
        raise SystemExit(f"No pude leer {ruta}")

    SALIDA_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(SALIDA_DIR / "hog.jpg"), demo_hog(img), [cv2.IMWRITE_JPEG_QUALITY, 90])
    demo_histograma(img, SALIDA_DIR / "histograma.png")
    print(f"Guardado {SALIDA_DIR}/hog.jpg y histograma.png")


if __name__ == "__main__":
    main()
