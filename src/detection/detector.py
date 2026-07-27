#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULO 2 - Deteccion con YOLOv8.

Fine-tuning de YOLOv8 sobre el dataset aumentado y funciones limpias de
train / predict / evaluate.

Uso:
    python detector.py train
    python detector.py predict  --source data/raw
    python detector.py evaluate
"""

import argparse
import os
import shutil

import cv2
import numpy as np

# Utilidades

# Al vivir en src/detection/, la raíz del repo está DOS niveles arriba.
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolver_data(dataset_yaml):
    """
    Si el data.yaml tiene un path relativo, genera una copia con el path
    absoluto (respecto al repo) para que ultralytics lo encuentre siempre.
    """
    import yaml
    with open(dataset_yaml, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    ruta = cfg.get("path", "")
    if not ruta or os.path.isabs(ruta):
        return dataset_yaml

    cfg["path"] = os.path.normpath(os.path.join(REPO_DIR, ruta))
    resuelto = os.path.join(REPO_DIR, ".data_resuelto.yaml")
    with open(resuelto, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True)
    return resuelto


def _detectar_dispositivo():
    """Elige el mejor dispositivo: CUDA (NVIDIA), MPS (Mac) o CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            nombre = torch.cuda.get_device_name(0)
            vram   = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[INFO] GPU detectada: {nombre} ({vram:.1f} GB VRAM)")
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            print("[INFO] Chip Apple detectado, usando MPS.")
            return "mps"
        print("[AVISO] Sin GPU, usando CPU.")
        return "cpu"
    except ImportError:
        return "cpu"


def _color_para_clase(class_id):
    """Color BGR estable y distinto por clase."""
    rng = np.random.default_rng(class_id * 97 + 13)
    return tuple(int(c) for c in rng.integers(60, 255, size=3))


# ENTRENAMIENTO

def train(dataset_yaml="data.yaml", model_size="n", epochs=100,
          project=None, name="tools", output_weights=None):
    """
    Fine-tuning de YOLOv8 sobre el dataset. Devuelve un dict de metricas.
    Hiperparametros optimizados para RTX 4050 4GB VRAM + dataset pequeno.
    """
    # Rutas por defecto ancladas a la raíz del repo (independientes del cwd).
    if project is None:
        project = os.path.join(REPO_DIR, "runs")
    if output_weights is None:
        output_weights = os.path.join(REPO_DIR, "models", "detection", "best.pt")
    from ultralytics import YOLO

    dataset_yaml = _resolver_data(dataset_yaml)
    device = _detectar_dispositivo()

    modelo = YOLO(f"yolov8{model_size}.pt")  # pesos preentrenados COCO

    hyperparams = dict(
        data=dataset_yaml,
        epochs=epochs,
        lr0=0.005,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5,
        box=7.5,
        cls=0.5,
        patience=20,
        imgsz=640,
        augment=False,      # usamos nuestro propio augmentation
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        # En Windows multiprocessing usa 'spawn' y cada worker re-picklea
        # el estado del padre (dataset+cache). Con workers=8 (default) y
        # cache=True el respawn de 'close_mosaic' puede reventar la RAM.
        workers=2,
    )

    if device == "cuda":
        hyperparams["batch"]  = 8      # seguro para 4GB VRAM con YOLOv8n
        hyperparams["cache"]  = True   # cachear dataset en RAM para ir mas rapido
        hyperparams["half"]   = True   # FP16: reduce VRAM a la mitad, misma precision
        print("[INFO] Modo GPU: batch=8, FP16 activado, cache en RAM.")
    elif device == "mps":
        hyperparams["batch"]  = 8
        hyperparams["cache"]  = True
        hyperparams["half"]   = False  # FP16 no aplica en MPS
        print("[INFO] Modo Mac (MPS): batch=8, cache en RAM.")
    else:
        hyperparams["batch"]  = 4
        hyperparams["cache"]  = False
        hyperparams["half"]   = False
        print("[AVISO] Modo CPU: batch=4, cache desactivado (sera lento).")

    resultados = modelo.train(**hyperparams)

    # Copiar mejores pesos a la ruta canonica.
    best = os.path.join(project, name, "weights", "best.pt")
    if os.path.exists(best):
        os.makedirs(os.path.dirname(output_weights), exist_ok=True)
        shutil.copy(best, output_weights)
        print(f"[OK] Mejores pesos guardados en: {output_weights}")

    metricas = {}
    try:
        metricas = {
            "mAP50":    float(resultados.box.map50),
            "mAP50_95": float(resultados.box.map),
        }
        print(f"[OK] mAP@0.5 = {metricas['mAP50']:.3f}")
    except Exception:
        pass
    return metricas

# PREDICCION


def predict(source, model_path=None, conf=0.4, iou=0.5, output_dir=None):
    """
    Ejecuta prediccion sobre una imagen, array numpy o carpeta.
    Devuelve lista de dicts con class_id, class_name, confidence,
    bbox_xyxy (pixeles) y bbox_xywhn (YOLO normalizado).
    """
    if model_path is None:
        model_path = os.path.join(REPO_DIR, "models", "detection", "best.pt")
    if output_dir is None:
        output_dir = os.path.join(REPO_DIR, "outputs", "predicciones")
    from ultralytics import YOLO

    device = _detectar_dispositivo()
    modelo = YOLO(model_path)
    os.makedirs(output_dir, exist_ok=True)

    resultados = modelo.predict(source=source, conf=conf, iou=iou,
                                device=device, verbose=False)
    detecciones = []
    for res in resultados:
        nombres = res.names
        img     = res.orig_img.copy()

        for box in res.boxes:
            cid       = int(box.cls[0])
            confianza = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            cx, cy, w, h    = [float(v) for v in box.xywhn[0]]

            det = {
                "class_id":   cid,
                "class_name": nombres.get(cid, str(cid)),
                "confidence": confianza,
                "bbox_xyxy":  [x1, y1, x2, y2],
                "bbox_xywhn": [cx, cy, w, h],
            }
            detecciones.append(det)

            color = _color_para_clase(cid)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            etq = f"{det['class_name']} {confianza:.2f}"
            cv2.putText(img, etq, (int(x1), max(int(y1) - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        nombre = os.path.basename(getattr(res, "path", "pred")) or "pred.jpg"
        nombre = os.path.splitext(nombre)[0] + "_pred.jpg"
        cv2.imwrite(os.path.join(output_dir, nombre), img)

    print(f"[OK] {len(detecciones)} detecciones. Visualizaciones en: {output_dir}")
    return detecciones

# EVALUACION

def evaluate(dataset_yaml="data.yaml", model_path=None, umbral=0.75):
    """
    Ejecuta YOLOv8 val sobre el split test e imprime metricas.
    """
    if model_path is None:
        model_path = os.path.join(REPO_DIR, "models", "detection", "best.pt")
    from ultralytics import YOLO

    dataset_yaml = _resolver_data(dataset_yaml)
    device = _detectar_dispositivo()
    if device == "mps":
        # en MPS el NMS de la validacion se pasa del limite de tiempo
        # y tumba el recall; en CPU la evaluacion es rapida igual
        device = "cpu"
    modelo = YOLO(model_path)
    print(f"[INFO] Evaluando sobre split TEST ({dataset_yaml})...")

    metricas = modelo.val(data=dataset_yaml, split="test", device=device,
                          plots=True, verbose=False)

    map50   = float(metricas.box.map50)
    map5095 = float(metricas.box.map)
    nombres = metricas.names

    print("\n" + "=" * 66)
    print("RESULTADOS DE EVALUACION (split test)")
    print("=" * 66)
    print(f"{'Clase':<24}{'mAP@0.5':>10}{'Precision':>12}{'Recall':>10}")
    print("-" * 66)

    resultado_clases = {}
    try:
        for i, cid in enumerate(metricas.box.ap_class_index):
            nombre = nombres.get(int(cid), str(cid))
            ap50   = float(metricas.box.ap50[i])
            p      = float(metricas.box.p[i])
            r      = float(metricas.box.r[i])
            resultado_clases[nombre] = {"mAP50": ap50, "precision": p, "recall": r}
            print(f"{nombre:<24}{ap50:>10.3f}{p:>12.3f}{r:>10.3f}")
    except Exception as exc:
        print(f"[WARN] No se pudieron leer metricas por clase: {exc}")

    print("-" * 66)
    print(f"{'GLOBAL mAP@0.5':<24}{map50:>10.3f}")
    print(f"{'GLOBAL mAP@0.5:0.95':<24}{map5095:>10.3f}")

    # Copiar los graficos (matriz de confusion, curvas) a outputs/metricas
    save_dir = str(getattr(metricas, "save_dir", ""))
    destino  = os.path.join(REPO_DIR, "outputs", "metricas")
    os.makedirs(destino, exist_ok=True)
    if save_dir and os.path.isdir(save_dir):
        for f in os.listdir(save_dir):
            if f.endswith((".png", ".jpg", ".csv")):
                shutil.copy(os.path.join(save_dir, f), os.path.join(destino, f))
        print(f"[OK] Graficos de metricas copiados a: {destino}/")

    print("\n" + "=" * 66)
    if map50 >= umbral:
        print(f"✅ PASSED  (mAP@0.5 = {map50:.3f} >= {umbral})")
    else:
        print(f"❌ FAILED — objetivo es {umbral}  (mAP@0.5 = {map50:.3f})")
    print("=" * 66)

    return {"mAP50": map50, "mAP50_95": map5095,
            "por_clase": resultado_clases, "passed": map50 >= umbral}

# CLI

def main():
    parser = argparse.ArgumentParser(description="Detector YOLOv8 de herramientas.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Entrenar el modelo.")
    p_train.add_argument("--data",       default="data.yaml")
    p_train.add_argument("--model_size", default="n", choices=["n", "s"])
    p_train.add_argument("--epochs",     type=int, default=100)

    p_pred = sub.add_parser("predict", help="Predecir sobre imagen/carpeta.")
    p_pred.add_argument("--source",     required=True)
    p_pred.add_argument("--model",      default=None)
    p_pred.add_argument("--conf",       type=float, default=0.4)
    p_pred.add_argument("--output_dir", default=None)

    p_eval = sub.add_parser("evaluate", help="Evaluar sobre split test.")
    p_eval.add_argument("--data",  default="data.yaml")
    p_eval.add_argument("--model", default=None)

    args = parser.parse_args()

    if args.cmd == "train":
        train(args.data, model_size=args.model_size, epochs=args.epochs)
    elif args.cmd == "predict":
        predict(args.source, model_path=args.model,
                conf=args.conf, output_dir=args.output_dir)
    elif args.cmd == "evaluate":
        evaluate(args.data, model_path=args.model)


if __name__ == "__main__":
    main()