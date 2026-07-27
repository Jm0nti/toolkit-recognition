#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Demo visual de la segmentación clásica (Otsu y GrabCut).

Toma unas fotos del dataset, segmenta con los dos métodos y guarda una
comparativa por foto en outputs/segmentacion/:

    original | máscara Otsu | overlay Otsu | overlay GrabCut

GrabCut se inicializa con los bounding boxes del detector YOLO entrenado,
así la segmentación queda integrada al pipeline.

Uso:
    python scripts/run_segmentation_demo.py            # 2 fotos por persona
    python scripts/run_segmentation_demo.py --n 4      # 4 por persona
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.classification.segmentation import mask_grabcut, mask_otsu, overlay_mask  # noqa: E402
from src.paths import DATA_RAW_DIR, DETECTION_MODELS, OUTPUTS_DIR, RUNS_DIR  # noqa: E402

SALIDA_DIR = OUTPUTS_DIR / "segmentacion"


def reducir(img: np.ndarray, lado_max: int = 1600) -> np.ndarray:
    """Reescala la foto para que GrabCut no tarde minutos en fotos de celular."""
    h, w = img.shape[:2]
    if max(h, w) <= lado_max:
        return img
    escala = lado_max / max(h, w)
    return cv2.resize(img, (int(w * escala), int(h * escala)),
                      interpolation=cv2.INTER_AREA)


def elegir_fotos(por_persona: int) -> list[Path]:
    """Las primeras N fotos de cada carpeta de data/raw."""
    fotos = []
    for carpeta in sorted(DATA_RAW_DIR.iterdir()):
        if carpeta.is_dir():
            imgs = sorted(p for p in carpeta.iterdir()
                          if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            fotos.extend(imgs[:por_persona])
    return fotos


def cargar_detector():
    from ultralytics import YOLO
    ruta = DETECTION_MODELS / "best.pt"
    if not ruta.exists():
        ruta = RUNS_DIR / "tools" / "weights" / "best.pt"
    return YOLO(str(ruta))


def panel(img: np.ndarray, ancho: int = 640) -> np.ndarray:
    """Reescala a un ancho fijo para armar la tira comparativa."""
    h, w = img.shape[:2]
    nh = int(h * ancho / w)
    return cv2.resize(img, (ancho, nh), interpolation=cv2.INTER_AREA)


def rotular(img: np.ndarray, texto: str) -> np.ndarray:
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (25, 25, 25), -1)
    cv2.putText(img, texto, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=2, help="fotos por persona")
    ap.add_argument("--conf", type=float, default=0.4, help="confianza del detector")
    args = ap.parse_args()

    SALIDA_DIR.mkdir(parents=True, exist_ok=True)
    modelo = cargar_detector()
    fotos = elegir_fotos(args.n)
    print(f"Segmentando {len(fotos)} fotos -> {SALIDA_DIR}")

    for ruta in fotos:
        img = cv2.imread(str(ruta))
        if img is None:
            print(f"  [aviso] no pude leer {ruta.name}, la salto")
            continue
        img = reducir(img)

        # método 1: Otsu sobre toda la imagen
        otsu = mask_otsu(img)

        # método 2: GrabCut por cada bbox del detector
        res = modelo.predict(source=img, conf=args.conf, verbose=False)[0]
        gc_total = np.zeros(img.shape[:2], dtype=np.uint8)
        for box in res.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            gc_total = cv2.bitwise_or(gc_total, mask_grabcut(img, (x1, y1, x2, y2)))

        tira = cv2.hconcat([
            rotular(panel(img), "original"),
            rotular(panel(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)), "mascara Otsu"),
            rotular(panel(overlay_mask(img, otsu)), "overlay Otsu"),
            rotular(panel(overlay_mask(img, gc_total, color=(80, 220, 100))),
                    f"GrabCut ({len(res.boxes)} bboxes)"),
        ])
        destino = SALIDA_DIR / f"{ruta.parent.name}_{ruta.stem}_seg.jpg"
        cv2.imwrite(str(destino), tira, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"  {destino.name}")

    print("Listo.")


if __name__ == "__main__":
    main()
