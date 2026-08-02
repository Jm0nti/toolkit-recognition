"""Extracción de características para los clasificadores clásicos.

Provee dos descriptores complementarios:

  * HOG (Histogram of Oriented Gradients) — captura forma y bordes
    a partir de la imagen en escala de grises. Ideal para siluetas de
    herramientas (contornos y ángulos de mangos, cabezas, etc.).
  * Histograma de color 3D en HSV — captura la distribución cromática,
    útil para distinguir herramientas con colores característicos
    (mangos plásticos rojos vs. metálicos, etc.).

Ambos devuelven vectores 1D numpy (float32) para ser usados directamente
por cualquier clasificador de scikit-learn.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import hog

from .preprocessing import resize_pad, to_grayscale


HOG_TARGET_SIZE = 128
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(16, 16),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    transform_sqrt=True,
    feature_vector=True,
)


def hog_features(img_bgr: np.ndarray, size: int = HOG_TARGET_SIZE) -> np.ndarray:
    """Devuelve el descriptor HOG del recorte (float32, 1D)."""
    canvas = resize_pad(img_bgr, size)
    gray = to_grayscale(canvas)
    vec = hog(gray, **HOG_PARAMS)
    return vec.astype(np.float32)


def color_histogram(img_bgr: np.ndarray,
                    bins: tuple[int, int, int] = (8, 8, 8),
                    color_space: str = "hsv") -> np.ndarray:
    """
    Histograma 3D de color aplanado y normalizado por norma L2 (norma = 1).

    HSV es más robusto a cambios de iluminación que BGR — recomendado
    para las herramientas fotografiadas en un cuarto no controlado.
    """
    if color_space == "hsv":
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        ranges = [0, 180, 0, 256, 0, 256]  # H tiene rango [0,179]
    else:  # bgr
        img = img_bgr
        ranges = [0, 256, 0, 256, 0, 256]

    hist = cv2.calcHist([img], [0, 1, 2], None, list(bins), ranges)
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)


def combined_features(img_bgr: np.ndarray) -> np.ndarray:
    """Concatena HOG + histograma de color HSV (útil para KNN u otro clasificador combinado)."""
    return np.concatenate([hog_features(img_bgr), color_histogram(img_bgr)])
