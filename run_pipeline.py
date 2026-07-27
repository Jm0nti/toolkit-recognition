#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODULO 3 - Orquestador del pipeline completo.

Flujo actual (la anotacion es MANUAL, no automatica):

    0) [MANUAL, fuera de este script]
       Anotas las imagenes a mano y exportas un archivo .ndjson de Ultralytics
       junto a las imagenes en data/raw/.
    1) Aumento de datos + splits  (augmentation_pipeline.py)
    2) Generacion / verificacion de data.yaml
    3) Entrenamiento YOLOv8        (detector.py)
    4) Evaluacion                  (detector.py)

Uso:
    python run_pipeline.py --model_size n --epochs 100

Nota: las rutas de entrada (imagenes y .ndjson) y el factor de aumento se
configuran DENTRO de augmentation_pipeline.py (bloque `class args`).
"""

import argparse
import os

import augmentation_pipeline
import detector


def generar_data_yaml(augmented_dir, ruta_salida="data.yaml"):
    """
    Verifica o genera data.yaml.

    - Si data.yaml ya existe (curado a mano), se respeta y se usa tal cual.
    - Si no existe, se genera a partir de classes.txt del dataset aumentado.
    """
    if os.path.exists(ruta_salida):
        print(f"[INFO] Usando data.yaml existente: {ruta_salida}")
        return ruta_salida

    ruta_classes = os.path.join(augmented_dir, "classes.txt")
    if not os.path.exists(ruta_classes):
        print(f"[ERROR] No existe {ruta_salida} ni {ruta_classes}. "
              f"Ejecuta primero el aumento o crea data.yaml a mano.")
        return None

    with open(ruta_classes, "r", encoding="utf-8") as fh:
        clases = [l.strip() for l in fh if l.strip()]

    # Ruta relativa para que el data.yaml sirva en cualquier maquina del equipo
    contenido = (
        f"path: {augmented_dir}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {len(clases)}\n"
        f"names: {clases}\n"
    )
    with open(ruta_salida, "w", encoding="utf-8") as fh:
        fh.write(contenido)
    print(f"[OK] data.yaml generado ({len(clases)} clases): {ruta_salida}")
    return ruta_salida


def main():
    parser = argparse.ArgumentParser(description="Orquestador del pipeline.")
    parser.add_argument("--augmented_dir", default="data/augmented",
                        help="Carpeta del dataset aumentado (salida del Modulo 1).")
    parser.add_argument("--data", default="data.yaml",
                        help="Ruta al data.yaml.")
    parser.add_argument("--model_size", default="n", choices=["n", "s"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--skip_augmentation", action="store_true",
                        help="Salta el Modulo 1 si el dataset ya esta aumentado.")
    args = parser.parse_args()

    print("\n########## PASO 0: ANOTACION MANUAL ##########")
    print("Recuerda: este pipeline asume que YA anotaste las imagenes a mano")
    print("y exportaste el .ndjson de Ultralytics junto a las imagenes.")
    print("Las rutas de entrada se configuran dentro de augmentation_pipeline.py.")

    print("\n########## PASO 1: AUMENTO DE DATOS ##########")
    if args.skip_augmentation:
        print("[INFO] Saltando aumento (--skip_augmentation).")
    else:
        # augmentation_pipeline.main() usa su propia configuracion interna
        # (bloque `class args`): input_dir, ndjson, output_dir, factor...
        augmentation_pipeline.main()

    print("\n########## PASO 2: VERIFICAR data.yaml ##########")
    data_yaml = generar_data_yaml(args.augmented_dir, args.data)
    if data_yaml is None:
        return

    print("\n########## PASO 3: ENTRENAMIENTO ##########")
    metricas_train = detector.train(data_yaml, model_size=args.model_size,
                                    epochs=args.epochs)

    print("\n########## PASO 4: EVALUACION ##########")
    metricas_eval = detector.evaluate(data_yaml, model_path="models/best.pt")

    print("\n########## RESUMEN FINAL ##########")
    print(f"Entrenamiento: {metricas_train}")
    print(f"mAP@0.5 test:  {metricas_eval['mAP50']:.3f}")
    print(f"Objetivo 0.75: "
          f"{'ALCANZADO ✅' if metricas_eval['passed'] else 'NO alcanzado ❌'}")
    print("\nPipeline finalizado.")


if __name__ == "__main__":
    main()
