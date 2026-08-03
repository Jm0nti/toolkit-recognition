# Reconocimiento de Herramientas Manuales

Pipeline completo de **visión artificial** sobre fotos cenitales de cajas de
herramientas: unificación de anotaciones colaborativas, aumento de datos,
**detección** con YOLOv8, **segmentación** (anotación poligonal + Otsu/GrabCut)
y **clasificación** con tres métodos ML complementarios (dos clásicos + una CNN). El repositorio cumple los cuatro ejes de la rúbrica
académica: *descripción del dataset, metodología, presentación de resultados y
métricas, análisis y conclusiones*.

---

## 1. Descripción del dataset

- **Origen.** Fotos tomadas con smartphone por cuatro colaboradores
  (`data/raw/jacob/`, `data/raw/juanl/`, `data/raw/juanma/`, `data/raw/tiago/`).
  Cada uno anotó su
  set en la plataforma Ultralytics y exportó un archivo NDJSON independiente.
  El script `src/data/merge_ndjson.py` los une con orden canónico de clases
  (ver §3.1).
- **Contexto.** Herramientas dentro de cajas o sobre superficies planas, con
  **oclusión parcial**, superposición y fondos caóticos.
- **Composición.** 10 clases (`metro, destornillador, martillo, pinzas,
  alicate, llave_inglesa, llave_combinada, brocha, espatula, llave_tubo`). El
  NDJSON unificado descarta cualquier clase fuera de este set (p. ej.
  `rodillo_pintura` de uno de los colaboradores).
- **Cantidad.** **36 imágenes originales** con **152 instancias anotadas** →
  **597 imágenes** tras el aumento (factor 12 + 30 % de mosaicos,
  `data/augmented/`) → **4.113 recortes** de bbox para clasificación
  (`outputs/ml_datasets/crops/`).
- **Particiones (post-aumento).** train / val / test = 413 / 116 / 68 imágenes
  (2909 / 807 / 442 instancias). Para clasificación: 2880 / 799 / 434 recortes.
- **Distribución.** El split por defecto es 70/20/10 estratificado por clase
  dominante. La distribución exacta por clase se imprime al final del Módulo 1
  y en el Módulo 4 al construir los recortes.
- **Limitaciones.**
  - Volumen bajo por clase (algunas <150 recortes en train antes del aumento).
  - Sesgo de iluminación (mismas condiciones dentro de cada colaborador).
  - Solo cuatro personas → variabilidad de estilo de foto acotada.
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
| 5 | Segmentación clásica | `src/classification/segmentation.py` | Otsu y GrabCut (inicializado con los bboxes del detector) |

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
| Segmentación | `segmentation.mask_otsu`, `segmentation.mask_grabcut` | Ver §4.5 y la justificación en §6 |
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

Los pesos entrenados quedan en `runs/tools/weights/best.pt`. Si existe
`models/detection/best.pt` el frontend lo usa primero; si no, cae solo
al de `runs/`.

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

### 4.5 Demo de segmentación clásica (Módulo 5)

```bash
python scripts/run_segmentation_demo.py          # 2 fotos por colaborador
python scripts/run_segmentation_demo.py --n 4
```

Por cada foto guarda en `outputs/segmentacion/` una tira comparativa:
`original | máscara Otsu | overlay Otsu | overlay GrabCut`. GrabCut se
inicializa con los bounding boxes del detector, así los dos métodos quedan
comparados sobre las mismas fotos reales del dataset.

---

## 5. Métricas y presentación de resultados

### 5.1 Módulo de detección (YOLOv8)

- **mAP@0.5** (métrica principal; objetivo académico ≥ 0.75).
- **mAP@0.5:0.95** (promedio a distintos IoU, más estricta).
- **Precision / Recall por clase**, matriz de confusión y curvas P/R.

Estos artefactos los produce Ultralytics automáticamente en `runs/tools/` y
se copian a `outputs/metricas/` al ejecutar `evaluate`.

**Resultados obtenidos** (YOLOv8n, 100 épocas, ~16.7 min de entrenamiento):

| Métrica | Validación (116 imgs, 807 inst) | Test (68 imgs, 442 inst) |
|---|---:|---:|
| **mAP@0.5** | **0.9535** | **0.9432** |
| mAP@0.5:0.95 | 0.8379 | 0.7961 |
| Precision | 0.8977 | 0.8669 |
| Recall | 0.9206 | 0.9204 |

Se supera con margen la meta académica de mAP@0.5 ≥ 0.75. Las tres peores
clases son `pinzas` (mAP@0.5 = 0.854), `llave_inglesa` (0.859) y `alicate`
(0.864) — coinciden con las tres peores del clasificador (ver §6).

### 5.2 Módulo de clasificación (Módulo 4)

Por cada uno de los tres modelos:

- **Accuracy** y **macro-F1** globales sobre el split test.
- **Precisión, Recall y F1** por clase (usando `classification_report` de
  sklearn con `zero_division=0`).
- **Matriz de confusión** normalizada por fila (recall visual por clase).
- **Tiempo de entrenamiento** y **tiempo de predicción** en el split test.

Y la tabla comparativa unificada (`comparison.md`) permite justificar por qué
un método le gana a otro en el análisis.

**Resultados obtenidos** (split test, N = 434 recortes; los mismos números que
`outputs/ml_reports/comparison.md`):

| Modelo | Accuracy | Macro-F1 | N test | Train (s) | Pred (s) |
|---|---:|---:|---:|---:|---:|
| hog_svm       | **0.8456** | **0.8241** | 434 |  17.90 | 3.54 |
| cnn           | 0.8203 | 0.8081 | 434 | 635.49 | 1.87 |
| color_hist_rf | 0.8065 | 0.7888 | 434 |   5.22 | 0.88 |

Los reportes por clase (P/R/F1) y matrices de confusión están en
`outputs/ml_reports/`.

---

## 6. Estructura del proyecto

```
toolkit-recognition/
├── README.md
├── requirements.txt
├── data.yaml                                # config YOLO
├── data/
│   ├── raw/
│   │   ├── jacob/ juanl/ juanma/ tiago/     # imágenes por colaborador
│   │   └── unified.ndjson                   # NDJSON unificado (salida M1)
│   └── augmented/                           # dataset YOLO (salida M2)
│       ├── images/{train,val,test}          # (ignoradas por git, se regeneran)
│       ├── labels/{train,val,test}
│       └── classes.txt
├── src/                                     # código fuente
│   ├── paths.py                             # rutas ancladas al repo
│   ├── data/
│   │   └── merge_ndjson.py                  # Módulo 1
│   ├── detection/
│   │   ├── augmentation_pipeline.py         # Módulo 2
│   │   └── detector.py                      # Módulo 3
│   └── classification/                      # Módulos 4 y 5
│       ├── preprocessing.py                 # gris, canales, bordes
│       ├── features.py                      # HOG, color histogram
│       ├── segmentation.py                  # Otsu + GrabCut (Módulo 5)
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
│   ├── run_ml_pipeline.py                   # M4 - todo el flujo
│   └── run_segmentation_demo.py             # M5 - comparativas de segmentación
├── outputs/
│   ├── predicciones/                        # detecciones del frontend (ignoradas)
│   ├── metricas/                            # gráficos YOLO
│   ├── segmentacion/                        # comparativas Otsu vs GrabCut
│   ├── ml_datasets/crops/{train,val,test}/  # recortes (ignorados, se regeneran)
│   └── ml_reports/                          # métricas + matrices + comparación
├── runs/                                    # logs y checkpoints de Ultralytics
│   └── tools/weights/best.pt                # pesos YOLO entrenados
└── web/index.html                           # frontend estático
```

Los modelos de clasificación entrenados (`models/classification/*`) y los
recortes se regeneran con los scripts del Módulo 4, por eso no van en git.

---