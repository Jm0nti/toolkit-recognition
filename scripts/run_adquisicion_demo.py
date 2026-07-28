#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Montaje de adquisición: una foto de cada colaborador.

Arma una tira con una foto real por persona (jacob, juanl, juanma, tiago)
para mostrar de dónde salieron los datos. Se guarda en outputs/adquisicion/.

Uso:
    python scripts/run_adquisicion_demo.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.paths import DATA_RAW_DIR, OUTPUTS_DIR  # noqa: E402

SALIDA_DIR = OUTPUTS_DIR / "adquisicion"
ALTO = 340


def _primera_foto(carpeta: Path):
    for p in sorted(carpeta.iterdir()):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            return p
    return None


def a_alto(img: np.ndarray, alto: int = ALTO) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * alto / h), alto), interpolation=cv2.INTER_AREA)


def rotular(img: np.ndarray, texto: str) -> np.ndarray:
    cv2.rectangle(img, (0, 0), (img.shape[1], 32), (25, 25, 25), -1)
    cv2.putText(img, texto, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (242, 184, 7), 2, cv2.LINE_AA)
    return img


def main() -> None:
    paneles = []
    for carpeta in sorted(DATA_RAW_DIR.iterdir()):
        if not carpeta.is_dir():
            continue
        foto = _primera_foto(carpeta)
        if foto is None:
            continue
        img = cv2.imread(str(foto))
        if img is None:
            continue
        paneles.append(rotular(a_alto(img), carpeta.name))

    if not paneles:
        raise SystemExit("No encontré fotos en data/raw/")

    montaje = cv2.hconcat(paneles)
    SALIDA_DIR.mkdir(parents=True, exist_ok=True)
    destino = SALIDA_DIR / "montaje.jpg"
    cv2.imwrite(str(destino), montaje, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"Guardado {destino} ({len(paneles)} colaboradores)")


if __name__ == "__main__":
    main()
