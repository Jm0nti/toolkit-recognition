#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULO 1 - Aumento de datos con Albumentations.

Recibe el dataset anotado y genera un dataset expandido con aumento clásico
y mosaico. Finalmente crea los splits train/val/test con la estructura que
espera YOLOv8.

FUENTES DE ENTRADA soportadas:
  1. Carpeta con imágenes + archivos .txt YOLO (flujo original)
  2. Carpeta de imágenes + archivo .ndjson de Ultralytics (flujo nuevo)
     En este caso las anotaciones se leen directamente del NDJSON sin
     necesitar convertirlas previamente.

Uso (flujo original con .txt):
    python augmentation_pipeline.py \\
        --input_dir data/annotated --output_dir data/augmented --factor 12

Uso (flujo nuevo con NDJSON):
    python augmentation_pipeline.py \\
        --input_dir data/raw --ndjson annotations.ndjson \\
        --output_dir data/augmented --factor 12

    Agrega --seg_mode para conservar polígonos completos (YOLOv8-seg).
    Por defecto convierte los polígonos a bboxes (detección).
"""

import json
import os
import random
import shutil
from collections import defaultdict

import cv2
import numpy as np
import albumentations as A

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
TARGET   = 640  # tamaño objetivo

# SECCIÓN 1 — Lectura de anotaciones

def poligono_a_bbox(coords: list[float]) -> tuple[float, float, float, float]:
    """Convierte polígono normalizado [x1,y1,...] → bbox YOLO (cx,cy,w,h)."""
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = max(0.0, min(1.0, (x_min + x_max) / 2))
    cy = max(0.0, min(1.0, (y_min + y_max) / 2))
    w  = max(0.001, min(1.0, x_max - x_min))
    h  = max(0.001, min(1.0, y_max - y_min))
    return cx, cy, w, h


def leer_yolo_txt(ruta_txt: str) -> tuple[list, list]:
    """Lee un .txt YOLO detección → (bboxes [[cx,cy,w,h]], labels [int])."""
    bboxes, labels = [], []
    if not os.path.exists(ruta_txt):
        return bboxes, labels
    with open(ruta_txt, "r", encoding="utf-8") as fh:
        for linea in fh:
            partes = linea.split()
            if len(partes) != 5:
                continue
            cid, cx, cy, w, h = partes
            bboxes.append([float(cx), float(cy), float(w), float(h)])
            labels.append(int(cid))
    return bboxes, labels


def escribir_yolo_txt(ruta_txt: str, bboxes: list, labels: list) -> None:
    """Escribe bboxes YOLO de detección en un .txt."""
    with open(ruta_txt, "w", encoding="utf-8") as fh:
        for (cx, cy, w, h), cid in zip(bboxes, labels):
            fh.write(f"{int(cid)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

# Carga desde NDJSON

def cargar_ndjson(ruta_ndjson: str) -> tuple[dict, dict]:
    """
    Lee el NDJSON de Ultralytics.

    Devuelve:
        meta          → dict del registro type='dataset'
        mapa_imagen   → {nombre_archivo: [(class_id, [coords...]), ...]}
    """
    meta        = {}
    mapa_imagen = {}

    with open(ruta_ndjson, "r", encoding="utf-8") as fh:
        for linea in fh:
            linea = linea.strip()
            if not linea:
                continue
            try:
                obj = json.loads(linea)
            except json.JSONDecodeError:
                continue

            if obj.get("type") == "dataset":
                meta = obj
            elif obj.get("type") == "image":
                nombre = obj.get("file", "")
                segs   = obj.get("annotations", {}).get("segments", [])
                anotaciones = []
                for seg in segs:
                    try:
                        cid   = int(seg[0])
                        coords = [float(v) for v in seg[1:]]
                        anotaciones.append((cid, coords))
                    except (ValueError, IndexError):
                        pass
                mapa_imagen[nombre] = anotaciones

    return meta, mapa_imagen


def _buscar_imagen(nombre_archivo: str, carpetas: list[str]) -> str | None:
    """Busca una imagen (por ruta relativa o basename) dentro de una lista de carpetas."""
    basename = os.path.basename(nombre_archivo)
    for carpeta in carpetas:
        candidatos = [
            os.path.join(carpeta, nombre_archivo),
            os.path.join(carpeta, basename),
        ]
        for ruta in candidatos:
            if os.path.exists(ruta):
                return ruta
    return None


def listar_pares_ndjson(
    carpetas_imgs: list[str] | str,
    mapa_imagen: dict,
    seg_mode: bool = False,
) -> list[tuple[str, str, list, list]]:
    """
    Construye la lista de pares (ruta_img, base, bboxes, labels)
    usando las anotaciones del NDJSON.

    `carpetas_imgs` puede ser una ruta (str) o una lista de rutas. La imagen
    se busca en cada carpeta por ruta relativa y por basename; se toma la
    primera coincidencia encontrada.

    Si seg_mode=False, convierte polígonos a bboxes (detección).
    Si seg_mode=True,  mantiene polígonos (segmentación — NO compatible
                        con el pipeline de augmentation actual que usa
                        BboxParams; se ignoran las coords extra).
    """
    if isinstance(carpetas_imgs, str):
        carpetas_imgs = [carpetas_imgs]

    pares = []
    for nombre_archivo, anotaciones in mapa_imagen.items():
        ruta_img = _buscar_imagen(nombre_archivo, carpetas_imgs)
        if ruta_img is None:
            print(f"[WARN] Imagen no encontrada en ninguna carpeta: {nombre_archivo}")
            continue

        bboxes, labels = [], []
        for cid, coords in anotaciones:
            cx, cy, w, h = poligono_a_bbox(coords)
            bboxes.append([cx, cy, w, h])
            labels.append(cid)

        base = os.path.splitext(os.path.basename(nombre_archivo))[0]
        pares.append((ruta_img, base, bboxes, labels))

    return pares


def listar_pares_txt(carpetas: list[str] | str) -> list[tuple[str, str, list, list]]:
    """
    Carga pares desde una o varias carpetas con imágenes + .txt YOLO.
    Devuelve lista de (ruta_img, base, bboxes, labels).
    """
    if isinstance(carpetas, str):
        carpetas = [carpetas]

    pares = []
    for carpeta in carpetas:
        if not os.path.isdir(carpeta):
            print(f"[WARN] Carpeta no encontrada: {carpeta}")
            continue
        for f in sorted(os.listdir(carpeta)):
            if not f.lower().endswith(IMG_EXTS):
                continue
            base     = os.path.splitext(f)[0]
            ruta_txt = os.path.join(carpeta, base + ".txt")
            if not os.path.exists(ruta_txt):
                continue
            ruta_img       = os.path.join(carpeta, f)
            bboxes, labels = leer_yolo_txt(ruta_txt)
            pares.append((ruta_img, base, bboxes, labels))
    return pares

# SECCIÓN 2 — Pipeline de augmentation

def _try_kwargs(clase, *variantes):
    """Instancia `clase` probando varios conjuntos de kwargs (compatibilidad)."""
    ultimo_error = None
    for kwargs in variantes:
        try:
            return clase(**kwargs)
        except (TypeError, ValueError) as exc:
            ultimo_error = exc
    raise ultimo_error


def construir_pipeline():
    """Crea el pipeline de aumento. Todas las transformaciones actualizan bbox."""
    transforms = [
        # Geométricas: críticas para fotos cenitales de herramientas
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15,
                           rotate_limit=45, p=0.6),
        _try_kwargs(
            A.RandomResizedCrop,
            {"size": (TARGET, TARGET), "scale": (0.7, 1.0), "p": 0.4},
            {"height": TARGET, "width": TARGET, "scale": (0.7, 1.0), "p": 0.4},
        ),

        # Fotométricas: simulan iluminación de taller
        A.RandomBrightnessContrast(brightness_limit=0.3,
                                   contrast_limit=0.3, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30,
                             val_shift_limit=20, p=0.5),
        A.CLAHE(clip_limit=2.0, p=0.3),
        _try_kwargs(
            A.ImageCompression,
            {"quality_range": (70, 100), "p": 0.3},
            {"quality_lower": 70, "quality_upper": 100, "p": 0.3},
        ),

        # Ruido: simula sensor de smartphone
        A.GaussNoise(var_limit=(10.0, 40.0), p=0.4),
        A.MotionBlur(blur_limit=5, p=0.2),
        A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2.0), p=0.2),

        # Oclusión: herramientas superpuestas
        _try_kwargs(
            A.CoarseDropout,
            {"num_holes_range": (1, 4), "hole_height_range": (10, 40),
             "hole_width_range": (10, 40), "p": 0.3},
            {"max_holes": 4, "max_height": 40, "max_width": 40, "p": 0.3},
        ),
    ]
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="yolo",
                                 label_fields=["class_labels"],
                                 min_visibility=0.4),
    )

# SECCIÓN 3 — Mosaico

def mosaico(pares: list, size: int = TARGET):
    """
    Combina 4 imágenes aleatorias en una rejilla size×size, fusionando bboxes.
    `pares` es lista de (ruta_img, base, bboxes, labels).
    """
    seleccion = random.sample(pares, 4) if len(pares) >= 4 else \
        [random.choice(pares) for _ in range(4)]
    mitad   = size // 2
    lienzo  = np.zeros((size, size, 3), dtype=np.uint8)
    offsets = [(0, 0), (mitad, 0), (0, mitad), (mitad, mitad)]
    out_bboxes, out_labels = [], []

    for (ruta_img, _base, bboxes, labels), (ox, oy) in zip(seleccion, offsets):
        img = cv2.imread(ruta_img)
        if img is None:
            continue
        img = cv2.resize(img, (mitad, mitad))
        lienzo[oy:oy + mitad, ox:ox + mitad] = img

        for (cx, cy, w, h), cid in zip(bboxes, labels):
            ncx = (ox + cx * mitad) / size
            ncy = (oy + cy * mitad) / size
            nw  = (w * mitad) / size
            nh  = (h * mitad) / size
            out_bboxes.append([ncx, ncy, nw, nh])
            out_labels.append(cid)

    return lienzo, out_bboxes, out_labels

# SECCIÓN 4 — Splits y guardado

def crear_estructura(base: str) -> None:
    """Crea images/{train,val,test} y labels/{train,val,test}."""
    for tipo in ("images", "labels"):
        for split in ("train", "val", "test"):
            os.makedirs(os.path.join(base, tipo, split), exist_ok=True)


def split_estratificado(
    items: list,
    ratios: tuple = (0.7, 0.2, 0.1),
) -> tuple[list, list, list]:
    """
    Divide (nombre, bboxes, labels) en train/val/test por clase dominante.
    """
    por_clase = defaultdict(list)
    for item in items:
        _n, _b, labels = item
        dominante = max(set(labels), key=labels.count) if labels else -1
        por_clase[dominante].append(item)

    train, val, test = [], [], []
    for _clase, grupo in por_clase.items():
        random.shuffle(grupo)
        n       = len(grupo)
        n_train = int(n * ratios[0])
        n_val   = int(n * ratios[1])
        train  += grupo[:n_train]
        val    += grupo[n_train:n_train + n_val]
        test   += grupo[n_train + n_val:]

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


def guardar_split(items: list, base: str, split: str) -> None:
    """Guarda imágenes y .txt en el split indicado."""
    for nombre, img, bboxes, labels in items:
        cv2.imwrite(os.path.join(base, "images", split, nombre + ".jpg"), img)
        escribir_yolo_txt(
            os.path.join(base, "labels", split, nombre + ".txt"),
            bboxes, labels,
        )

# SECCIÓN 5 — Principal

def main():
    # CONFIGURACIÓN — Edita estos valores según tu proyecto

    class args:
        # Lista de carpetas con imágenes originales. Cada colaborador puede
        # tener su propia subcarpeta; el pipeline busca cada imagen del
        # NDJSON en todas ellas (por ruta relativa y por basename).
        input_dirs   = [
            "data/raw/juanl",
            "data/raw/juanma",
            "data/raw/tiago",
        ]
        ndjson       = "data/raw/unified.ndjson"         # NDJSON unificado
        output_dir   = "data/augmented"         # carpeta de salida del dataset
        factor       = 12                       # imágenes generadas por imagen original
        mosaic_ratio = 0.30                     # fracción de imágenes vía mosaico
        seed         = 42

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Cargar pares según la fuente
    if args.ndjson:
        if not os.path.exists(args.ndjson):
            print(f"[ERROR] NDJSON no encontrado: {args.ndjson}")
            return
        print(f"[INFO] Fuente: NDJSON ({args.ndjson})")
        print(f"[INFO] Buscando imágenes en: {args.input_dirs}")
        meta, mapa_imagen = cargar_ndjson(args.ndjson)
        nombres_clase: dict = meta.get("class_names", {})
        print(f"[INFO] Clases detectadas: {nombres_clase}")
        pares = listar_pares_ndjson(args.input_dirs, mapa_imagen)
    else:
        print(f"[INFO] Fuente: carpetas con .txt YOLO ({args.input_dirs})")
        pares = listar_pares_txt(args.input_dirs)
        nombres_clase = _leer_classes_txt(args.input_dirs)

    if not pares:
        print(f"[ERROR] No se encontraron pares imagen/anotación.")
        return

    print(f"[INFO] {len(pares)} imágenes anotadas encontradas. Factor={args.factor}")

    # Aumento clásico
    pipeline  = construir_pipeline()
    generados = []  # (nombre, img_bgr, bboxes, labels)

    for ruta_img, base, bboxes, labels in pares:
        img = cv2.imread(ruta_img)
        if img is None:
            print(f"[WARN] No se pudo leer: {ruta_img}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Original siempre entra al dataset
        generados.append((f"{base}_orig",
                          cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                          bboxes, labels))

        for k in range(args.factor):
            try:
                res = pipeline(image=img_rgb, bboxes=bboxes,
                               class_labels=labels)
            except Exception as exc:
                print(f"[WARN] Fallo aumento en {base} #{k}: {exc}")
                continue
            aug_img = cv2.cvtColor(res["image"], cv2.COLOR_RGB2BGR)
            generados.append((f"{base}_aug{k}", aug_img,
                              list(res["bboxes"]),
                              [int(c) for c in res["class_labels"]]))

    # Mosaicos 
    total_objetivo = len(pares) * args.factor
    n_mosaico      = int(total_objetivo * args.mosaic_ratio)

    for m in range(n_mosaico):
        img, bboxes, labels = mosaico(pares)
        generados.append((f"mosaic_{m}", img, bboxes, labels))

    print(f"[INFO] Total imágenes generadas: {len(generados)} "
          f"(incluye {n_mosaico} mosaicos)")

    # Split estratificado 
    items_meta = [(n, b, l) for (n, _img, b, l) in generados]
    train_m, val_m, test_m = split_estratificado(items_meta)
    nombres_train = {n for n, _b, _l in train_m}
    nombres_val   = {n for n, _b, _l in val_m}

    crear_estructura(args.output_dir)
    train_full, val_full, test_full = [], [], []
    for nombre, img, bboxes, labels in generados:
        if nombre in nombres_train:
            train_full.append((nombre, img, bboxes, labels))
        elif nombre in nombres_val:
            val_full.append((nombre, img, bboxes, labels))
        else:
            test_full.append((nombre, img, bboxes, labels))

    guardar_split(train_full, args.output_dir, "train")
    guardar_split(val_full,   args.output_dir, "val")
    guardar_split(test_full,  args.output_dir, "test")

    # Guardar classes.txt 
    ruta_classes = os.path.join(args.output_dir, "classes.txt")
    if isinstance(nombres_clase, dict):
        with open(ruta_classes, "w", encoding="utf-8") as fh:
            for k in sorted(int(k) for k in nombres_clase):
                fh.write(nombres_clase[str(k)] + "\n")
    elif isinstance(nombres_clase, list) and nombres_clase:
        with open(ruta_classes, "w", encoding="utf-8") as fh:
            fh.write("\n".join(nombres_clase) + "\n")

    # Copiar classes.txt desde la primera input_dir que lo tenga (modo TXT)
    if not args.ndjson:
        for carpeta in args.input_dirs:
            src = os.path.join(carpeta, "classes.txt")
            if os.path.exists(src):
                shutil.copy(src, ruta_classes)
                break

    # Estadísticas
    dist_train = defaultdict(int)
    for _n, _img, _b, labels in train_full:
        for cid in labels:
            dist_train[int(cid)] += 1

    nc_list = _leer_classes_desde_ruta(ruta_classes)

    print("\n" + "=" * 60)
    print("ESTADÍSTICAS DEL DATASET")
    print("=" * 60)
    print(f"Train: {len(train_full)}  |  Val: {len(val_full)}  |  "
          f"Test: {len(test_full)}")
    print("\nDistribución de clases en TRAIN (por class_id):")
    # Recorre todas las clases canónicas para que las que tienen 0
    # instancias también sean visibles.
    ids_a_reportar = sorted(set(range(len(nc_list))) | set(dist_train.keys()))
    for cid in ids_a_reportar:
        count = dist_train.get(cid, 0)
        etq   = nc_list[cid] if cid < len(nc_list) else str(cid)
        if count == 0:
            marca = "  <-- SIN INSTANCIAS"
        elif count < 20:
            marca = "  <-- POCAS (<20)"
        else:
            marca = ""
        print(f"  [{cid:2d}] {etq:<24} {count:>5}{marca}")

    faltantes = [c for c in range(len(nc_list)) if dist_train.get(c, 0) < 20]
    if faltantes:
        print("\n[AVISO] Desbalance: clases con <20 instancias en train.")
        print("        Considera anotar/aumentar más ejemplos de esas clases.")

    print(f"\n[OK] Dataset listo en: {args.output_dir}")


# Helpers de lectura de clases

def _leer_classes_txt(carpetas: list[str] | str):
    """Lee classes.txt desde la primera carpeta que lo tenga; devuelve dict {str_id: nombre}."""
    if isinstance(carpetas, str):
        carpetas = [carpetas]
    for carpeta in carpetas:
        ruta = os.path.join(carpeta, "classes.txt")
        if os.path.exists(ruta):
            with open(ruta, "r", encoding="utf-8") as fh:
                lineas = [l.strip() for l in fh if l.strip()]
            return {str(i): nombre for i, nombre in enumerate(lineas)}
    return {}


def _leer_classes_desde_ruta(ruta: str) -> list:
    """Lee un classes.txt y devuelve lista ordenada de nombres."""
    if not os.path.exists(ruta):
        return []
    with open(ruta, "r", encoding="utf-8") as fh:
        return [l.strip() for l in fh if l.strip()]


if __name__ == "__main__":
    main()