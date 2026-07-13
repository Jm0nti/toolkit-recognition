#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frontend web del detector.

Se prende con:
    python app.py
y se abre http://localhost:8000 en el navegador.
"""

import base64
import io
import os
import time

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

from detector import _detectar_dispositivo, _color_para_clase

MODEL_PATH = "models/best.pt"
SALIDA_DIR = "outputs/predicciones"

app = FastAPI(title="Detector de herramientas")

# el modelo se carga una sola vez, en la primera peticion
_modelo = None
_device = None


def cargar_modelo():
    global _modelo, _device
    if _modelo is None:
        from ultralytics import YOLO
        _device = _detectar_dispositivo()
        _modelo = YOLO(MODEL_PATH)
    return _modelo


@app.get("/")
def inicio():
    return FileResponse(os.path.join("web", "index.html"))


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

    # guardar copia en outputs/predicciones
    os.makedirs(SALIDA_DIR, exist_ok=True)
    nombre = time.strftime("%Y%m%d_%H%M%S") + "_pred.jpg"
    cv2.imwrite(os.path.join(SALIDA_DIR, nombre), salida)

    # imagen anotada en base64 para mostrarla directo en la pagina
    ok, buf = cv2.imencode(".jpg", salida, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_b64 = base64.b64encode(buf).decode() if ok else ""

    conteo = {}
    for d in detecciones:
        conteo[d["clase"]] = conteo.get(d["clase"], 0) + 1

    return {
        "imagen": f"data:image/jpeg;base64,{img_b64}",
        "detecciones": detecciones,
        "conteo": conteo,
        "tiempo_ms": tiempo_ms,
        "guardada_en": os.path.join(SALIDA_DIR, nombre),
    }


if __name__ == "__main__":
    print("Detector de herramientas -> http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
