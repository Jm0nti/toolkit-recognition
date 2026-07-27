"""Operaciones de preprocesamiento clásicas: gris, canales, bordes.

Estas funciones cubren tres de los requisitos de la rúbrica de VA:
  * Conversión a escala de grises.
  * Separación de canales de color (BGR y HSV).
  * Detección de bordes (Canny y Sobel).

Todas trabajan sobre imágenes BGR (formato de OpenCV) uint8 de tamaño arbitrario.
"""
from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(img_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 → gris uint8."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def split_bgr(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Separa los canales azul, verde, rojo."""
    b, g, r = cv2.split(img_bgr)
    return b, g, r


def split_hsv(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BGR → HSV y separa canales matiz, saturación, valor."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return h, s, v


def edges_canny(img_bgr: np.ndarray,
                low: int = 80,
                high: int = 180,
                blur_ksize: int = 5) -> np.ndarray:
    """Detección de bordes con Canny (previo blur para suprimir ruido)."""
    gray = to_grayscale(img_bgr)
    if blur_ksize and blur_ksize >= 3:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    return cv2.Canny(gray, low, high)


def edges_sobel(img_bgr: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Magnitud del gradiente Sobel en X e Y, normalizada a uint8."""
    gray = to_grayscale(img_bgr).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = cv2.magnitude(gx, gy)
    mag = np.clip(mag / (mag.max() + 1e-9) * 255.0, 0, 255).astype(np.uint8)
    return mag


def resize_pad(img_bgr: np.ndarray, size: int = 128) -> np.ndarray:
    """Redimensiona conservando aspecto, rellenando con negro hasta size×size."""
    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    escala = size / max(h, w)
    nh, nw = max(1, int(h * escala)), max(1, int(w * escala))
    redim = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    lienzo = np.zeros((size, size, 3), dtype=np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    lienzo[y0:y0 + nh, x0:x0 + nw] = redim
    return lienzo
