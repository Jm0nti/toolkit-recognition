"""Segmentación clásica: umbral de Otsu y GrabCut.

Cubre el requisito de segmentación de la rúbrica con dos enfoques:
  * Otsu: umbral global automático sobre la imagen en gris, con limpieza
    morfológica. Rápido, funciona bien cuando el fondo es uniforme (cartón).
  * GrabCut: segmentación iterativa inicializada con el bounding box del
    detector, separa la herramienta del fondo dentro de su caja.

Todas trabajan sobre imágenes BGR (formato de OpenCV) uint8.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.classification.preprocessing import to_grayscale


def mask_otsu(img_bgr: np.ndarray,
              blur_ksize: int = 5,
              open_ksize: int = 5,
              close_ksize: int = 9) -> np.ndarray:
    """Máscara binaria por Otsu (0 fondo, 255 objeto), con limpieza morfológica.

    Se asume que las herramientas ocupan menos área que el fondo: si el
    umbral deja más de la mitad de la imagen en blanco, se invierte.
    """
    gray = to_grayscale(img_bgr)
    if blur_ksize and blur_ksize >= 3:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # el objeto debe ser la clase minoritaria
    if np.count_nonzero(mask) > mask.size / 2:
        mask = cv2.bitwise_not(mask)

    # abrir quita puntos sueltos, cerrar rellena huecos dentro de la herramienta
    if open_ksize >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_ksize >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def mask_grabcut(img_bgr: np.ndarray,
                 rect_xyxy: tuple[int, int, int, int],
                 iters: int = 5) -> np.ndarray:
    """Máscara binaria por GrabCut inicializado con un bbox (x1, y1, x2, y2).

    Todo lo de afuera del rect se marca como fondo seguro; adentro GrabCut
    decide qué es herramienta y qué es fondo.
    """
    x1, y1, x2, y2 = rect_xyxy
    h, w = img_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return np.zeros((h, w), dtype=np.uint8)

    mask = np.zeros((h, w), dtype=np.uint8)
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)
    rect = (x1, y1, x2 - x1, y2 - y1)
    cv2.grabCut(img_bgr, mask, rect, bg_model, fg_model, iters,
                cv2.GC_INIT_WITH_RECT)

    # GC_FGD y GC_PR_FGD son objeto; el resto fondo
    binaria = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0)
    return binaria.astype(np.uint8)


def overlay_mask(img_bgr: np.ndarray,
                 mask: np.ndarray,
                 color: tuple[int, int, int] = (0, 200, 255),
                 alpha: float = 0.45) -> np.ndarray:
    """Pinta la máscara semitransparente sobre la imagen y dibuja su contorno."""
    salida = img_bgr.copy()
    capa = np.zeros_like(salida)
    capa[mask > 0] = color
    salida = cv2.addWeighted(salida, 1.0, capa, alpha, 0)
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(salida, contornos, -1, color, 2)
    return salida
