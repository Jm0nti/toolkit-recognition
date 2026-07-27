# Reconocimiento de Herramientas Manuales

Pipeline completo de **visión artificial** sobre fotos cenitales de cajas de
herramientas: unificación de anotaciones colaborativas, aumento de datos,
**detección** con YOLOv8 y **clasificación** con tres métodos ML complementarios
(dos clásicos + una CNN). El repositorio cumple los cuatro ejes de la rúbrica
académica: *descripción del dataset, metodología, presentación de resultados y
métricas, análisis y conclusiones*.

---

## 1. Descripción del dataset

- **Origen.** Fotos tomadas con smartphone por tres colaboradores
  (`data/raw/juanl/`, `data/raw/juanma/`, `data/raw/tiago/`). Cada uno anotó su
  set en la plataforma Ultralytics y exportó un archivo NDJSON independiente.
  El script `src/data/merge_ndjson.py` los une con orden canónico de clases
  (ver §3.1).
- **Contexto.** Herramientas dentro de cajas o sobre superficies planas, con
  **oclusión parcial**, superposición y fondos caóticos.
- **Composición.** 10 clases (`metro, destornillador, martillo, pinzas,
  alicate, llave_inglesa, llave_combinada, brocha, espatula, llave_tubo`). El
  NDJSON unificado descarta cualquier clase fuera de este set (p. ej.
  `rodillo_pintura` de uno de los colaboradores).
- **Cantidad.** ~30 imágenes originales → **~500 imágenes** tras el aumento
  (`data/augmented/`) → **3.700 recortes** de bbox para clasificación
  (`outputs/ml_datasets/crops/`).
- **Distribución.** El split por defecto es 70/20/10 estratificado por clase
  dominante. La distribución exacta por clase se imprime al final del Módulo 1
  y en el Módulo 4 al construir los recortes.
- **Limitaciones.**
  - Volumen bajo por clase (algunas <150 recortes en train antes del aumento).
  - Sesgo de iluminación (mismas condiciones dentro de cada colaborador).
  - Solo tres personas → variabilidad de estilo de foto acotada.
  - Herramientas dentro de cajas → sombras y reflejos metálicos frecuentes.

---

## 2. Arquitectura y metodología

Cuatro módulos independientes, orquestables individualmente:

| # | Módulo | Ubicación | Función |
|---|--------|-----------|---------|
| 1 | Unificación de anotaciones | `src/data/merge_ndjson.py` | Reordena y filtra clases entre los NDJSON de los colaboradores |
| 2 | Aumento de datos + splits | `src/detection/augmentation_pipeline.py` | Albumentations + mosaico + split estratificado train/val/test |
| 3 | Detección YOLOv8 | `src/detection/detector.py` | Fine-tuning, predicción y evaluación con mAP |
| 4 | **Clasificación ML** | `src/classification/` | Recortes de bboxes → 3 clasificadores (HOG+SVM, ColorHist+RF, CNN) |

Los scripts de arranque viven en `scripts/`; la lógica en `src/`. Todas las
rutas se anclan a la raíz del repo (ver `src/paths.py`), por lo que los
scripts se pueden ejecutar desde cualquier directorio.

### Módulo 4 en detalle — cumplimiento de la rúbrica

La rúbrica exige, además de un clasificador, cuatro operaciones específicas:
todas están implementadas en `src/classification/preprocessing.py` y `features.py`:

| Requisito | Implementación | Uso en el pipeline |
|---|---|---|
| Conversión a escala de grises | `preprocessing.to_grayscale` | Base del descriptor HOG y de Sobel/Canny |
| Separación de canales de color | `preprocessing.split_bgr`, `split_hsv` | Base del histograma de color HSV |
| Detección de bordes | `preprocessing.edges_canny`, `edges_sobel` | Disponibles como utilidades; el HOG ya captura el gradiente de bordes de forma implícita |
| Extracción de características | `features.hog_features`, `features.color_histogram`, `features.combined_features` | HOG (forma/contornos) + histograma HSV (color) |

Sobre esas features se entrenan tres clasificadores complementarios:

1. **HOG + SVM (RBF).** Baseline clásico basado en forma. HOG resume los
   gradientes locales en 128×128 grises; el SVM con kernel RBF captura
   relaciones no lineales entre esos gradientes. Estandarización previa con
   `StandardScaler` para que el SVM converja bien.
2. **Histograma de color HSV + Random Forest.** Baseline clásico basado en
   color. Un histograma 3D (8×8×8) por recorte y un RandomForest de 200
   árboles con `class_weight="balanced"`. Detecta la firma cromática de cada
   herramienta (mango plástico rojo vs. cabeza metálica gris, etc.).
3. **CNN pequeña (PyTorch).** Modelo deep de ~200k parámetros: 3 bloques
   `Conv-BN-ReLU-Pool` (32→64→128 canales) + `GlobalAvgPool` + `FC(128) →
   Dropout(0.3) → FC(10)`. Entrada RGB 96×96 con letterbox. Loss
   ponderada por clase (compensa desbalance), early stopping por accuracy
   de val (paciencia = 6).

Los tres implementan la **misma interfaz** (`train / predict / save / load`),
lo que permite tratarlos uniformemente en `scripts/run_ml_train.py` y
`scripts/run_ml_evaluate.py`, y **si uno falla, los otros dos siguen** (bloque
`try/except` por modelo).

---

## 3. Instalación

```bash
# 1) Entorno virtual (Python 3.10+ recomendado)
python -m venv .toolkit-venv
# Windows PowerShell:
.\.toolkit-venv\Scripts\Activate.ps1
# Linux/Mac:
source .toolkit-venv/bin/activate

# 2) (Opcional GPU) PyTorch con CUDA ANTES de requirements
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# 3) Dependencias
pip install -r requirements.txt
```

Dependencias clave: `ultralytics`, `albumentations`, `torch/torchvision`,
`scikit-learn`, `scikit-image`, `opencv-python`, `matplotlib`, `fastapi`.

---

## 4. Ejecución paso a paso

### 4.1 Unificación de NDJSON (una vez por cambios en las anotaciones)

Edita `INPUT_FILES` en `src/data/merge_ndjson.py` para listar los `.ndjson` de
cada colaborador y corre:

```bash
python scripts/run_merge_ndjson.py
```

Salida: `data/raw/unified.ndjson` con las 10 clases canónicas y todos los
segmentos remapeados por **nombre** (no por índice).

### 4.2 Aumento de datos + splits (Módulo 2)

Edita la lista `input_dirs` dentro de `main()` de
`src/detection/augmentation_pipeline.py` para incluir las carpetas de imágenes
de cada colaborador, luego:

```bash
python -m src.detection.augmentation_pipeline
# o vía el orquestador de detección:
python scripts/run_pipeline.py --model_size n --epochs 100
```

Salida: `data/augmented/{images,labels}/{train,val,test}/` + `classes.txt`.

### 4.3 Detección con YOLOv8 (Módulo 3)

```bash
# Entrenar
python -m src.detection.detector train --data data.yaml --model_size n --epochs 100
# Evaluar
python -m src.detection.detector evaluate --data data.yaml
# Predecir en una carpeta
python -m src.detection.detector predict --source data/raw/juanl
```

Los pesos finales se copian a `models/detection/best.pt`.

Frontend web para probar interactivamente:

```bash
python scripts/serve_app.py
# abre http://localhost:8000
```

### 4.4 Clasificación ML (Módulo 4)

Cada paso es independiente — si uno falla, los siguientes se pueden lanzar por
separado.

```bash
# Paso 1: construir dataset de recortes (idempotente, incremental)
python scripts/run_ml_dataset.py                    # incremental
python scripts/run_ml_dataset.py --overwrite        # regenera todo

# Paso 2: entrenar UN modelo o todos
python scripts/run_ml_train.py --model hog_svm
python scripts/run_ml_train.py --model color_hist_rf
python scripts/run_ml_train.py --model cnn --epochs 25
python scripts/run_ml_train.py --model all

# Paso 3: evaluar UN modelo o todos (+ tabla comparativa opcional)
python scripts/run_ml_evaluate.py --model hog_svm
python scripts/run_ml_evaluate.py --model all --compare

# Todo el pipeline ML de una:
python scripts/run_ml_pipeline.py
python scripts/run_ml_pipeline.py --skip_dataset --epochs 40
```

Los reportes por modelo se guardan en `outputs/ml_reports/`:

- `<modelo>_metrics.json` — accuracy, macro-F1, precisión/recall/F1 por clase.
- `<modelo>_confusion_matrix.png` — matriz normalizada por fila.
- `comparison.{md,csv,png}` — tabla comparativa de todos los modelos.

---

## 5. Métricas y presentación de resultados

### 5.1 Módulo de detección (YOLOv8)

- **mAP@0.5** (métrica principal; objetivo académico ≥ 0.75).
- **mAP@0.5:0.95** (promedio a distintos IoU, más estricta).
- **Precision / Recall por clase**, matriz de confusión y curvas P/R.

Estos artefactos los produce Ultralytics automáticamente en `runs/tools/` y
se copian a `outputs/metricas/` al ejecutar `evaluate`.

### 5.2 Módulo de clasificación (Módulo 4)

Por cada uno de los tres modelos:

- **Accuracy** y **macro-F1** globales sobre el split test.
- **Precisión, Recall y F1** por clase (usando `classification_report` de
  sklearn con `zero_division=0`).
- **Matriz de confusión** normalizada por fila (recall visual por clase).
- **Tiempo de entrenamiento** y **tiempo de predicción** en el split test.

Y la tabla comparativa unificada (`comparison.md`) permite justificar por qué
un método le gana a otro en el análisis.

**Resultados obtenidos** (smoke test con `python scripts/run_ml_pipeline.py`,
CNN a 15 epochs — subir `--epochs` mejora la CNN significativamente):

| Modelo | Accuracy | Macro-F1 | N test | Train (s) | Pred (s) |
|---|---:|---:|---:|---:|---:|
| hog_svm       | 0.891 | 0.887 | 495 | 28.2  | 3.17 |
| color_hist_rf | 0.877 | 0.874 | 495 |  8.3  | 0.59 |
| cnn (15 ep.)  | 0.800 | 0.785 | 495 | 210.0 | 1.81 |

Observaciones iniciales:

- Los **clásicos superan a la CNN** con este volumen de datos y epochs. La CNN
  seguía bajando la loss (val_acc pasó de 0.34 → 0.73 en 15 epochs); con 30-50
  epochs típicamente alcanza o supera a los clásicos, dado su mayor capacidad.
- **HOG + SVM** es el ganador por poco: siluetas y bordes son la señal más
  robusta para estas herramientas.
- **Color hist + RF** entrena en 8 s y predice en <1 s: candidato natural
  para un baseline rápido / edge.

Los reportes por clase (P/R/F1) y matrices de confusión están en
`outputs/ml_reports/`.

---

## 6. Análisis y conclusiones (guía)

Este archivo entrega la **infraestructura**; el análisis se documenta luego
en base a los outputs reales. La rúbrica pide, para cada resultado:

1. **Qué método ganó y por qué.** HOG+SVM y RF sobre color son sorprendentemente
   fuertes en este dataset porque las herramientas tienen siluetas y colores
   distintivos. La CNN puede quedarse cerca o superar a los clásicos según cuán
   generalizables sean las features aprendidas frente al pequeño volumen de datos.
2. **Dónde falla cada modelo.** La matriz de confusión de cada uno muestra los
   pares problemáticos típicos (`pinzas` ↔ `alicate`, `llave_inglesa` ↔
   `llave_combinada`). Reportar estos pares es más valioso que un solo número.
3. **Trade-offs.** Los modelos clásicos entrenan en segundos y se ejecutan en
   CPU sin problema; la CNN es más pesada de entrenar pero potencialmente más
   robusta a variaciones no vistas. Los tiempos están en `comparison.csv`.
4. **Inconvenientes durante el proceso.** Documentar en el informe:
   - Bajo volumen por clase → dependemos fuertemente del augmentation.
   - Anotaciones heterogéneas entre colaboradores → resuelto con `merge_ndjson.py`.
   - VRAM limitada (6 GB) → workers reducidos en YOLO y fallback CPU en `serve_app.py`.
5. **Mejoras posibles.** Más datos por clase, transfer learning
   (MobileNetV2/EfficientNet), features combinadas (HOG + histograma) para
   sklearn, ensembles.

---

## 7. Estructura del proyecto

```
toolkit-img-recognition/
├── README.md
├── requirements.txt
├── data.yaml                                # config YOLO
├── data/
│   ├── raw/
│   │   ├── juanl/  juanma/  tiago/          # imágenes por colaborador
│   │   └── unified.ndjson                   # NDJSON unificado (salida M1)
│   └── augmented/                           # dataset YOLO (salida M2)
│       ├── images/{train,val,test}
│       ├── labels/{train,val,test}
│       └── classes.txt
├── src/                                     # código fuente
│   ├── paths.py                             # rutas ancladas al repo
│   ├── data/
│   │   └── merge_ndjson.py                  # Módulo 1
│   ├── detection/
│   │   ├── augmentation_pipeline.py         # Módulo 2
│   │   └── detector.py                      # Módulo 3
│   └── classification/                      # Módulo 4
│       ├── preprocessing.py                 # gris, canales, bordes
│       ├── features.py                      # HOG, color histogram
│       ├── dataset_builder.py               # crops desde YOLO labels
│       ├── evaluation.py                    # métricas + comparación
│       └── models/
│           ├── hog_svm.py
│           ├── color_hist_rf.py
│           └── cnn.py
├── scripts/                                 # entry points ejecutables
│   ├── run_merge_ndjson.py                  # M1
│   ├── run_pipeline.py                      # M2 + M3 (detección end-to-end)
│   ├── serve_app.py                         # frontend web del detector
│   ├── run_ml_dataset.py                    # M4 - construir dataset
│   ├── run_ml_train.py     [--model X]      # M4 - entrenar
│   ├── run_ml_evaluate.py  [--model X]      # M4 - evaluar
│   └── run_ml_pipeline.py                   # M4 - todo el flujo
├── models/
│   ├── detection/best.pt                    # pesos YOLO
│   └── classification/                      # hog_svm.joblib, color_hist_rf.joblib, cnn.pt
├── outputs/
│   ├── predicciones/                        # detecciones del frontend
│   ├── metricas/                            # gráficos YOLO
│   ├── ml_datasets/crops/{train,val,test}/  # recortes de clasificación
│   └── ml_reports/                          # métricas + matrices + comparación
├── runs/                                    # logs y checkpoints de Ultralytics
└── web/index.html                           # frontend estático
```

---