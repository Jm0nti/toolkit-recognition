#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Descarga el "6 tool dataset" de Ultralytics y lo deja en formato YOLO.

Lee el NDJSON de dataset-masivo/ (que trae las anotaciones y las URLs de
las imágenes), descarga las imágenes y escribe:

    data/masivo/images/{train,val,test}/*.jpg
    data/masivo/labels/{train,val,test}/*.txt
    data/masivo/data.yaml

Las URLs del NDJSON están firmadas y VENCEN (campo Expires); si fallan,
hay que re-exportar el NDJSON desde platform.ultralytics.com.

Uso:
    python scripts/run_dataset_masivo.py --limit 300   # muestra rápida
    python scripts/run_dataset_masivo.py               # todo (~1.8 GB)
"""
import argparse
import concurrent.futures as cf
import json
import ssl
import sys
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.paths import DATA_DIR, REPO_ROOT  # noqa: E402

NDJSON = REPO_ROOT / "dataset-masivo" / "6-tool-datasetyolov11.ndjson"
SALIDA = DATA_DIR / "masivo"

# macOS a veces no encuentra los certificados de Python; uso un contexto laxo
# solo para este CDN de ultralytics.
_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE


def leer_ndjson():
    """Devuelve (class_names, registros de imagen con anotaciones)."""
    clases, registros = None, []
    with open(NDJSON, encoding="utf-8") as fh:
        for linea in fh:
            linea = linea.strip()
            if not linea:
                continue
            o = json.loads(linea)
            if o.get("type") == "dataset":
                nombres = o.get("class_names", {})
                clases = [nombres[str(i)] for i in range(len(nombres))]
            elif o.get("type") == "image":
                registros.append(o)
    return clases, registros


def descargar(reg, im_dir, lb_dir):
    """Baja una imagen y escribe su .txt YOLO. Devuelve True si quedó bien."""
    nombre = Path(reg["file"]).name
    destino = im_dir / nombre
    if not destino.exists():
        try:
            req = urllib.request.Request(reg["url"], headers={"User-Agent": "curl/8"})
            datos = urllib.request.urlopen(req, timeout=30, context=_CTX).read()
            destino.write_bytes(datos)
        except Exception:
            return False

    lineas = []
    for b in reg.get("annotations", {}).get("boxes", []):
        cid, cx, cy, w, h = b
        lineas.append(f"{int(cid)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    (lb_dir / (destino.stem + ".txt")).write_text("\n".join(lineas) + "\n" if lineas else "",
                                                  encoding="utf-8")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0,
                    help="máximo de imágenes por split (0 = todas)")
    ap.add_argument("--workers", type=int, default=12, help="descargas en paralelo")
    args = ap.parse_args()

    if not NDJSON.exists():
        raise SystemExit(f"No encuentro {NDJSON}")

    clases, registros = leer_ndjson()
    print(f"{len(registros)} imágenes en el NDJSON, clases: {clases}")

    # organizar por split (y recortar si piden muestra)
    por_split = {"train": [], "val": [], "test": []}
    for r in registros:
        sp = r.get("split", "train")
        if sp in por_split:
            por_split[sp].append(r)
    if args.limit:
        # muestra proporcionada: --limit manda en train, val y test van más chicos
        topes = {"train": args.limit,
                 "val": max(50, args.limit // 4),
                 "test": max(25, args.limit // 8)}
        for sp in por_split:
            por_split[sp] = por_split[sp][:topes[sp]]

    total_ok, total_fail = 0, 0
    for sp, regs in por_split.items():
        im_dir = SALIDA / "images" / sp
        lb_dir = SALIDA / "labels" / sp
        im_dir.mkdir(parents=True, exist_ok=True)
        lb_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{sp}] descargando {len(regs)} imágenes...")
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            resultados = list(ex.map(lambda r: descargar(r, im_dir, lb_dir), regs))
        ok = sum(resultados)
        total_ok += ok
        total_fail += len(regs) - ok
        print(f"[{sp}] {ok}/{len(regs)} ok")

    # data.yaml para entrenar con ultralytics
    yaml = [f"path: {SALIDA}", "train: images/train", "val: images/val",
            "test: images/test", "", "names:"]
    yaml += [f"  {i}: {n}" for i, n in enumerate(clases)]
    (SALIDA / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")

    print(f"\nListo: {total_ok} ok, {total_fail} fallidas.")
    print(f"Config de entrenamiento: {SALIDA / 'data.yaml'}")
    if total_fail:
        print("Si fallaron muchas, el NDJSON pudo vencer (campo Expires): "
              "re-exportarlo desde platform.ultralytics.com.")


if __name__ == "__main__":
    main()
