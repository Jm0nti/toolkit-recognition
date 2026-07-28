#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Demo visual del preprocesamiento clásico.

Toma una foto real y muestra en una tira las operaciones de la rúbrica:

    original | gris | canal H (HSV) | bordes Canny | bordes Sobel

Guarda el resultado en outputs/preprocesamiento/ para usarlo en la página
y en el informe.

Uso:
    python scripts/run_preprocessing_demo.py
    python scripts/run_preprocessing_demo.py --img data/raw/tiago/6S.jpg
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.classification.preprocessing import (  # noqa: E402
    edges_canny, edges_sobel, split_hsv, to_grayscale,
)
from src.paths import DATA_RAW_DIR, OUTPUTS_DIR  # noqa: E402

SALIDA_DIR = OUTPUTS_DIR / "preprocesamiento"


def _primera_foto() -> Path:
    for carpeta in sorted(DATA_RAW_DIR.iterdir()):
        if carpeta.is_dir():
            for p in sorted(carpeta.iterdir()):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    return p
    raise SystemExit("No encontré fotos en data/raw/")


def panel(img: np.ndarray, ancho: int = 420) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (ancho, int(h * ancho / w)), interpolation=cv2.INTER_AREA)


def rotular(img: np.ndarray, texto: str) -> np.ndarray:
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (25, 25, 25), -1)
    cv2.putText(img, texto, (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def gris3(g: np.ndarray) -> np.ndarray:
    """Gris de 1 canal -> 3 canales para poder concatenar."""
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--img", type=str, default="", help="ruta de la foto a usar")
    args = ap.parse_args()

    ruta = Path(args.img) if args.img else _primera_foto()
    img = cv2.imread(str(ruta))
    if img is None:
        raise SystemExit(f"No pude leer {ruta}")

    # reescalo para que las operaciones sean rápidas y la tira quepa
    img = panel(img, 600)
    h, _, _ = split_hsv(img)

    tira = cv2.hconcat([
        rotular(panel(img.copy()), "original"),
        rotular(panel(gris3(to_grayscale(img))), "gris"),
        rotular(panel(gris3(h)), "canal H (HSV)"),
        rotular(panel(gris3(edges_canny(img))), "bordes Canny"),
        rotular(panel(gris3(edges_sobel(img))), "bordes Sobel"),
    ])

    SALIDA_DIR.mkdir(parents=True, exist_ok=True)
    destino = SALIDA_DIR / f"{ruta.stem}_preproc.jpg"
    cv2.imwrite(str(destino), tira, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Guardado {destino}")


if __name__ == "__main__":
    main()
