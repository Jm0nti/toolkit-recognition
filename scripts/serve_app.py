#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frontend web del detector.

Se prende con:
    python scripts/serve_app.py
y se abre http://localhost:8000 en el navegador.
"""

import base64
import csv
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
from src.paths import (  # noqa: E402
    DETECTION_MODELS, OUTPUTS_DIR, PREDICTIONS_DIR, RUNS_DIR, WEB_DIR,
)

SALIDA_DIR = str(PREDICTIONS_DIR)
os.makedirs(SALIDA_DIR, exist_ok=True)

# carpetas con las graficas de los dos modelos (deteccion y clasificacion)
METRICAS_DIR = OUTPUTS_DIR / "metricas"
ML_REPORTS_DIR = OUTPUTS_DIR / "ml_reports"
RUNS_TOOLS_DIR = RUNS_DIR / "tools"

app = FastAPI(title="Detector de herramientas")
# servir las imagenes guardadas para poder mostrarlas en el historial
app.mount("/predicciones", StaticFiles(directory=SALIDA_DIR), name="predicciones")

# servir las graficas de metricas y las evidencias del pipeline
for _url, _carpeta in [("/metricas-img", METRICAS_DIR),
                       ("/ml-img", ML_REPORTS_DIR),
                       ("/runs-img", RUNS_TOOLS_DIR),
                       ("/out", OUTPUTS_DIR)]:
    if _carpeta.exists():
        app.mount(_url, StaticFiles(directory=str(_carpeta)), name=_url.strip("/"))

# ── Registro de modelos ─────────────────────────────────────
# Cada modelo entrenado se ofrece en la página para poder compararlos.
# Solo aparecen los best.pt que existen en disco.
def _candidatos_modelos():
    cands = [
        ("mio", "El mío · 10 clases", RUNS_TOOLS_DIR / "weights" / "best.pt"),
        ("real6", "6-tool · dataset real", DETECTION_MODELS / "real6" / "best.pt"),
    ]
    listados = {c[0] for c in cands}
    # además cualquier otro models/detection/<nombre>/best.pt
    if DETECTION_MODELS.exists():
        for sub in sorted(DETECTION_MODELS.iterdir()):
            w = sub / "best.pt"
            if sub.is_dir() and w.exists() and sub.name not in listados:
                cands.append((sub.name, sub.name, w))
    return [(i, lbl, p) for i, lbl, p in cands if p.exists()]


_modelos_cache = {}   # id -> (YOLO, device)


def cargar_modelo(model_id=None):
    """Devuelve (id, modelo, device). Carga y cachea el modelo pedido.

    Si no existe ningún modelo, devuelve (None, None, None).
    """
    cands = _candidatos_modelos()
    if not cands:
        return None, None, None
    ids = [c[0] for c in cands]
    if model_id not in ids:
        model_id = ids[0]                 # por defecto, el primero (el mío)
    if model_id in _modelos_cache:
        modelo, device = _modelos_cache[model_id]
        return model_id, modelo, device

    ruta = next(p for i, _, p in cands if i == model_id)
    from ultralytics import YOLO
    import torch

    modelo = YOLO(str(ruta))
    device = _detectar_dispositivo()

    # Permite forzar CPU vía env var (VRAM ocupada por otro proceso, etc.)
    if os.environ.get("FORCE_CPU", "").lower() in ("1", "true", "yes"):
        device = "cpu"
    elif device == "cuda":
        # warm-up: si la GPU no tiene VRAM libre, caigo a CPU sin romper
        try:
            modelo.to("cuda")
            modelo.predict(source=np.zeros((32, 32, 3), dtype=np.uint8),
                           device="cuda", verbose=False)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                device = "cpu"
                modelo = YOLO(str(ruta))
            else:
                raise

    _modelos_cache[model_id] = (modelo, device)
    print(f"[INFO] modelo '{model_id}' cargado ({ruta}) en {device}.")
    return model_id, modelo, device


@app.get("/")
def inicio():
    return FileResponse(str(WEB_DIR / "index.html"))


@app.post("/detectar")
async def detectar(imagen: UploadFile = File(...), conf: float = 0.4,
                   modelo: str = None):
    datos = await imagen.read()
    arr = cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return JSONResponse({"error": "No pude leer la imagen."}, status_code=400)

    mid, red, device = cargar_modelo(modelo)
    if red is None:
        return JSONResponse(
            {"error": "No hay ningún modelo disponible. Entrena o coloca un best.pt."},
            status_code=503)

    t0 = time.time()
    resultados = red.predict(source=arr, conf=conf, iou=0.5,
                             device=device, verbose=False)
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
        "modelo": mid,
    }
    with open(os.path.join(SALIDA_DIR, base + ".json"), "w", encoding="utf-8") as fh:
        json.dump(registro, fh, ensure_ascii=False)

    # ademas la imagen en base64 para mostrarla al instante sin recargar
    ok, buf = cv2.imencode(".jpg", salida, [cv2.IMWRITE_JPEG_QUALITY, 90])
    inmediata = f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}" if ok else registro["imagen"]

    return {**registro, "imagen": inmediata,
            "guardada_en": os.path.join(SALIDA_DIR, base + ".jpg")}


@app.get("/modelos")
def modelos():
    """Modelos de detección disponibles para el selector de la página."""
    return {"modelos": [{"id": i, "nombre": lbl} for i, lbl, _ in _candidatos_modelos()]}


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


# ── Métricas de los modelos ────────────────────────────────
# Fuentes donde buscar una grafica, en orden de preferencia.
_FUENTES_IMG = [(METRICAS_DIR, "/metricas-img"), (RUNS_TOOLS_DIR, "/runs-img")]


def _buscar_grafica(nombre: str, titulo: str):
    """Devuelve {titulo, url} si la grafica existe en alguna fuente, si no None."""
    for carpeta, url_base in _FUENTES_IMG:
        if (carpeta / nombre).exists():
            return {"titulo": titulo, "url": f"{url_base}/{nombre}"}
    return None


def _metricas_deteccion() -> dict:
    d = {"curvas": [], "batches": []}
    csv_path = RUNS_TOOLS_DIR / "results.csv"
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as fh:
            filas = list(csv.DictReader(fh))
        if filas:
            ult = {k.strip(): v for k, v in filas[-1].items()}
            d["precision"] = float(ult["metrics/precision(B)"])
            d["recall"] = float(ult["metrics/recall(B)"])
            d["map50"] = float(ult["metrics/mAP50(B)"])
            d["map5095"] = float(ult["metrics/mAP50-95(B)"])
            d["epochs"] = int(float(ult["epoch"]))

    curvas = [
        ("PR_curve.png", "Precisión-Recall"),
        ("P_curve.png", "Precisión vs confianza"),
        ("R_curve.png", "Recall vs confianza"),
        ("F1_curve.png", "F1 vs confianza"),
        ("confusion_matrix_normalized.png", "Matriz de confusión (norm.)"),
        ("results.png", "Curvas de entrenamiento"),
    ]
    d["curvas"] = [g for n, t in curvas if (g := _buscar_grafica(n, t))]
    batches = [("val_batch0_pred.jpg", "Predicciones val (lote 0)"),
               ("val_batch1_pred.jpg", "Predicciones val (lote 1)")]
    d["batches"] = [g for n, t in batches if (g := _buscar_grafica(n, t))]
    return d


def _metricas_clasificacion() -> dict:
    modelos, matrices = [], []
    etiquetas = {"hog_svm": "HOG + SVM", "color_hist_rf": "Color hist + RF", "cnn": "CNN"}
    for nombre, titulo in etiquetas.items():
        jp = ML_REPORTS_DIR / f"{nombre}_metrics.json"
        if jp.exists():
            with open(jp, encoding="utf-8") as fh:
                m = json.load(fh)
            modelos.append({
                "nombre": titulo,
                "accuracy": m.get("accuracy"),
                "macro_f1": m.get("macro_f1"),
                "train_s": m.get("duracion_train_s"),
                "pred_s": m.get("duracion_pred_s"),
                "n_test": m.get("n_test"),
            })
        cm = ML_REPORTS_DIR / f"{nombre}_confusion_matrix.png"
        if cm.exists():
            matrices.append({"titulo": titulo, "url": f"/ml-img/{cm.name}"})

    comparativa = None
    if (ML_REPORTS_DIR / "comparison.png").exists():
        comparativa = "/ml-img/comparison.png"
    return {"modelos": modelos, "matrices": matrices, "comparativa": comparativa}


@app.get("/metricas")
def metricas():
    """Números y gráficas de los dos modelos, para la pestaña de métricas."""
    return {"deteccion": _metricas_deteccion(),
            "clasificacion": _metricas_clasificacion()}


def _listar_imgs(carpeta: Path, url_base: str) -> list:
    """URLs de las imagenes de una carpeta de evidencia (ordenadas)."""
    if not carpeta.exists():
        return []
    exts = (".jpg", ".jpeg", ".png")
    return [f"{url_base}/{p.name}" for p in sorted(carpeta.iterdir())
            if p.suffix.lower() in exts]


@app.get("/evidencia")
def evidencia():
    """Imágenes de cada etapa del pipeline, para la pestaña 'El proyecto'."""
    return {
        "adquisicion": _listar_imgs(OUTPUTS_DIR / "adquisicion", "/out/adquisicion"),
        "preprocesamiento": _listar_imgs(OUTPUTS_DIR / "preprocesamiento", "/out/preprocesamiento"),
        "segmentacion": _listar_imgs(OUTPUTS_DIR / "segmentacion", "/out/segmentacion"),
        "caracteristicas": _listar_imgs(OUTPUTS_DIR / "caracteristicas", "/out/caracteristicas"),
    }


if __name__ == "__main__":
    print("Detector de herramientas -> http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
