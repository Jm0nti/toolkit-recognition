#!/usr/bin/env python3
"""Unifica varios .ndjson de Ultralytics (segmentación) a un orden canónico de clases.

Cada archivo de entrada puede tener las clases en distinto orden y/o incluir
clases que no forman parte del conjunto canónico. Este script:

  * Re-mapea el índice de cada segmento al índice canónico según el NOMBRE de la clase.
  * Descarta segmentos cuyas clases no aparecen en el conjunto canónico.
  * Concatena todas las líneas "image" en un único .ndjson con una sola cabecera "dataset".

Configuración: edita INPUT_FILES y OUTPUT_FILE abajo y ejecuta:
    python merge_ndjson.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configuración: edita esta sección para agregar/quitar ndjson a unificar.    #
# Las rutas son relativas al directorio del script (o absolutas).             #
# --------------------------------------------------------------------------- #
INPUT_FILES: list[str] = [
    "data/raw/juanma/hand-tools.ndjson",
    "data/raw/tiago/herramientas.ndjson",
    "data/raw/juanl/annotations.ndjson",
    "data/raw/jacob/vision.ndjson"
]
OUTPUT_FILE: str = "data/raw/unified.ndjson"
DATASET_NAME: str = "toolkit-unified"
DATASET_TASK: str = "segment"
# --------------------------------------------------------------------------- #

CANONICAL_CLASSES: dict[int, str] = {
    0: "metro",
    1: "destornillador",
    2: "martillo",
    3: "pinzas",
    4: "alicate",
    5: "llave_inglesa",
    6: "llave_combinada",
    7: "brocha",
    8: "espatula",
    9: "llave_tubo",
}
NAME_TO_CANONICAL: dict[str, int] = {name: idx for idx, name in CANONICAL_CLASSES.items()}


def build_remap(class_names: dict) -> tuple[dict[int, int], list[str]]:
    """Devuelve (remap src_idx -> canonical_idx, lista de nombres descartados)."""
    remap: dict[int, int] = {}
    dropped: list[str] = []
    for src_idx_str, name in class_names.items():
        src_idx = int(src_idx_str)
        canonical = NAME_TO_CANONICAL.get(name)
        if canonical is None:
            dropped.append(name)
        else:
            remap[src_idx] = canonical
    return remap, dropped


def process_file(path: Path, seen_files: dict[str, str]) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        lines = [ln for ln in (line.strip() for line in f) if ln]

    if not lines:
        print(f"[{path.name}] archivo vacío, se omite", file=sys.stderr)
        return []

    header = json.loads(lines[0])
    if header.get("type") != "dataset":
        raise ValueError(f"{path}: la primera línea no es un header 'dataset'")

    remap, dropped = build_remap(header.get("class_names", {}))
    if dropped:
        print(f"[{path.name}] clases descartadas (fuera del canónico): {dropped}", file=sys.stderr)

    kept: list[dict] = []
    dropped_segments = 0
    for line in lines[1:]:
        obj = json.loads(line)
        if obj.get("type") != "image":
            continue

        file_name = obj.get("file")
        if file_name in seen_files:
            print(
                f"[{path.name}] imagen duplicada '{file_name}' "
                f"(ya provista por {seen_files[file_name]}) — se omite",
                file=sys.stderr,
            )
            continue

        segments = obj.get("annotations", {}).get("segments", []) or []
        remapped: list[list] = []
        for seg in segments:
            src_cls = int(seg[0])
            new_cls = remap.get(src_cls)
            if new_cls is None:
                dropped_segments += 1
                continue
            remapped.append([new_cls, *seg[1:]])

        obj.setdefault("annotations", {})["segments"] = remapped
        seen_files[file_name] = path.name
        kept.append(obj)

    print(
        f"[{path.name}] {len(kept)} imágenes conservadas, "
        f"{dropped_segments} segmentos descartados",
        file=sys.stderr,
    )
    return kept


def main() -> None:
    # Rutas relativas se resuelven contra la raíz del repo (dos niveles arriba
    # de src/data/), no contra el cwd.
    repo_root = Path(__file__).resolve().parents[2]
    inputs = [(repo_root / p).resolve() for p in INPUT_FILES]
    output = (repo_root / OUTPUT_FILE).resolve()

    missing = [p for p in inputs if not p.is_file()]
    if missing:
        for p in missing:
            print(f"ERROR: no existe el archivo de entrada: {p}", file=sys.stderr)
        sys.exit(1)

    seen_files: dict[str, str] = {}
    all_images: list[dict] = []
    for inp in inputs:
        all_images.extend(process_file(inp, seen_files))

    header = {
        "type": "dataset",
        "task": DATASET_TASK,
        "name": DATASET_NAME,
        "class_names": {str(k): v for k, v in CANONICAL_CLASSES.items()},
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for img in all_images:
            f.write(json.dumps(img, ensure_ascii=False) + "\n")

    print(f"OK: {len(all_images)} imágenes escritas en {output}")


if __name__ == "__main__":
    main()
