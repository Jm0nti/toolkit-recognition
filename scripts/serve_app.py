#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frontend web del detector.

Se prende con:
    python scripts/serve_app.py
y se abre http://localhost:8000 en el navegador.
"""

import base64
import glob
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Añade la raíz del repo al sys.path para que 'src' sea importable.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.detection.detector import _detectar_dispositivo, _color_para_clase  # noqa: E402
from src.paths import DETECTION_MODELS, PREDICTIONS_DIR, WEB_DIR  # noqa: E402

MODEL_PATH = str(DETECTION_MODELS / "best.pt")
SALIDA_DIR = str(PREDICTIONS_DIR)
os.makedirs(SALIDA_DIR, exist_ok=True)

app = FastAPI(title="Detector de herramientas")
# servir las imagenes guardadas para poder mostrarlas en el historial
app.mount("/predicciones", StaticFiles(directory=SALIDA_DIR), name="predicciones")

# el modelo se carga una sola vez, en la primera peticion
_modelo = None
_device = None


def cargar_modelo():
    global _modelo, _device
    if _modelo is not None:
        return _modelo

    from ultralytics import YOLO
    import torch

    _modelo = YOLO(MODEL_PATH)
    _device = _detectar_dispositivo()

    # Permite forzar CPU vía env var (útil si la VRAM está ocupada por
    # otro proceso: entrenamientos abiertos, navegador, IDE, etc.)
    if os.environ.get("FORCE_CPU", "").lower() in ("1", "true", "yes"):
        _device = "cpu"
        print("[INFO] FORCE_CPU=1 -> inferencia en CPU.")
        return _modelo

    # Intento de warm-up en el device elegido: si la GPU está sin VRAM
    # libre, hago fallback a CPU sin romper el request.
    if _device == "cuda":
        try:
            _modelo.to("cuda")
            # ping mínimo para provocar la asignación real de VRAM
            dummy = np.zeros((32, 32, 3), dtype=np.uint8)
            _modelo.predict(source=dummy, device="cuda", verbose=False)
            print("[INFO] Modelo cargado en GPU (cuda).")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" in str(exc).lower():
                print(f"[AVISO] GPU sin VRAM disponible ({exc.__class__.__name__}). "
                      f"Cayendo a CPU. Cierra otros procesos con nvidia-smi si quieres GPU.")
                torch.cuda.empty_cache()
                _device = "cpu"
                _modelo = YOLO(MODEL_PATH)  # recargar limpio en CPU
            else:
                raise
    return _modelo


@app.get("/")
def inicio():
    return FileResponse(str(WEB_DIR / "index.html"))


@app.post("/detectar")
async def detectar(imagen: UploadFile = File(...), conf: float = 0.4):
    datos = await imagen.read()
    arr = cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return JSONResponse({"error": "No pude leer la imagen."}, status_code=400)

    modelo = cargar_modelo()
    t0 = time.time()
    resultados = modelo.predict(source=arr, conf=conf, iou=0.5,
                                device=_device, verbose=False)
    tiempo_ms = int((time.time() - t0) * 1000)

    res = resultados[0]
    nombres = res.names
    salida = arr.copy()
    detecciones = []

    # grosor de linea acorde al tamano de la foto
    grosor = max(2, salida.shape[1] // 500)
    escala = max(0.6, salida.shape[1] / 1600)

    for box in res.boxes:
        cid = int(box.cls[0])
        confianza = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        clase = nombres.get(cid, str(cid))
        detecciones.append({"clase": clase, "confianza": confianza})

        color = _color_para_clase(cid)
        cv2.rectangle(salida, (x1, y1), (x2, y2), color, grosor)
        etiqueta = f"{clase} {confianza:.2f}"
        (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
        cv2.rectangle(salida, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(salida, etiqueta, (x1 + 3, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, escala, (20, 20, 20), grosor, cv2.LINE_AA)

    conteo = {}
    for d in detecciones:
        conteo[d["clase"]] = conteo.get(d["clase"], 0) + 1

    # guardar la imagen y sus datos (para el historial)
    base = time.strftime("%Y%m%d_%H%M%S") + "_pred"
    cv2.imwrite(os.path.join(SALIDA_DIR, base + ".jpg"), salida)
    registro = {
        "imagen": f"/predicciones/{base}.jpg",
        "detecciones": detecciones,
        "conteo": conteo,
        "tiempo_ms": tiempo_ms,
        "fecha": time.strftime("%d/%m/%Y %H:%M"),
    }
    with open(os.path.join(SALIDA_DIR, base + ".json"), "w", encoding="utf-8") as fh:
        json.dump(registro, fh, ensure_ascii=False)

    # ademas la imagen en base64 para mostrarla al instante sin recargar
    ok, buf = cv2.imencode(".jpg", salida, [cv2.IMWRITE_JPEG_QUALITY, 90])
    inmediata = f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}" if ok else registro["imagen"]

    return {**registro, "imagen": inmediata,
            "guardada_en": os.path.join(SALIDA_DIR, base + ".jpg")}


@app.get("/historial")
def historial(limite: int = 24):
    """Lista las ultimas predicciones guardadas, de la mas nueva a la mas vieja."""
    archivos = sorted(glob.glob(os.path.join(SALIDA_DIR, "*.json")), reverse=True)
    items = []
    for ruta in archivos[:limite]:
        try:
            with open(ruta, encoding="utf-8") as fh:
                items.append(json.load(fh))
        except Exception:
            pass
    return {"items": items}


if __name__ == "__main__":
    print("Detector de herramientas -> http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
