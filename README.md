# Deteccion de Herramientas Manuales en Cajas de Herramientas

Pipeline de **deteccion de objetos** para identificar herramientas manuales en
fotos cenitales (tomadas desde arriba con smartphone) de cajas de herramientas
desordenadas, con oclusion parcial y fondos caoticos.

El sistema esta dividido en **3 modulos independientes** + orquestador:

| Modulo | Archivo | Funcion |
|--------|---------|---------|
| 1 | `augmentation_pipeline.py` | Aumento de datos (Albumentations + mosaico) desde NDJSON + splits |
| 2 | `detector.py` | Fine-tuning, prediccion y evaluacion con YOLOv8 |
| 3 | `run_pipeline.py` | Orquestador end-to-end |

> **Nota:** la anotacion se hace **manualmente** (no hay auto-anotacion). El
> pipeline parte de un archivo NDJSON de Ultralytics con las anotaciones
> exportadas a mano.

**Meta academica:** `mAP@0.5 ≥ 0.75` en el conjunto de test con oclusion parcial.

Clases actuales (definidas en `data.yaml`): `metro, destornillador, martillo,
pinzas, alicate`. Puedes cambiarlas editando `data.yaml` y re-anotando.

---

## 1. Requisitos del sistema

- **Python 3.10 o superior** (se usan type hints `list[float]`, `tuple[...]`).
- **GPU NVIDIA con CUDA (recomendado)** para entrenar en minutos. El codigo
  detecta la GPU, activa **FP16** y ajusta `batch=8` (seguro para 4 GB VRAM,
  p.ej. RTX 4050).
- **CPU (fallback)**: funciona pero es lento; el codigo baja a `batch=4` y
  desactiva cache automaticamente.
- ~1 GB de espacio para pesos de YOLOv8 + dataset aumentado.
- Sistemas probados: Windows 11 y Linux.

---

## 2. Instalacion

```bash
# 1) Crear entorno virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 2) (GPU) Instalar PyTorch con CUDA ANTES de requirements
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# 3) Instalar dependencias
pip install -r requirements.txt
```

> Si no tienes GPU, omite el paso 2 e instala solo `pip install -r requirements.txt`.

---

## 3. Como agregar tus imagenes y anotarlas

### 3.1 Colocar las imagenes

1. Crea la carpeta `data/raw/`.
2. Copia ahi tus fotos `.jpg` / `.png` (20–100 imagenes cenitales).
3. Consejos: camara paralela al suelo, buena iluminacion (variar condiciones
   ayuda), incluye herramientas superpuestas y fondos reales de taller.

### 3.2 Anotar manualmente y exportar NDJSON

La anotacion es **manual**. Anota las herramientas (bounding box o poligono) con
una herramienta que exporte el **formato NDJSON de Ultralytics** y guarda el
archivo como `data/raw/annotations.ndjson`.

Estructura esperada del NDJSON (una linea JSON por registro):

```jsonl
{"type": "dataset", "class_names": {"0": "metro", "1": "destornillador", ...}}
{"type": "image", "file": "IMG_001.jpg", "annotations": {"segments": [[0, x1, y1, x2, y2, ...], ...]}}
```

- El registro `type: "dataset"` define el mapeo `class_id → nombre`.
- Cada `type: "image"` lista sus `segments`: el primer valor es el `class_id`
  y el resto son coordenadas **normalizadas** (poligono).
- El Modulo 1 convierte automaticamente cada poligono a bounding box YOLO
  (`poligono_a_bbox`). No necesitas convertir nada a mano.

> Si en su lugar tienes archivos `.txt` YOLO clasicos, el Modulo 1 tambien los
> soporta: pon `ndjson = None` en la configuracion (ver 4.1) y deja los `.txt`
> junto a las imagenes.

---

## 4. Ejecutar el pipeline

### 4.1 Configurar el Modulo 1

A diferencia del resto, `augmentation_pipeline.py` se configura **editando el
bloque `class args` dentro de `main()`** (no por linea de comandos):

```python
class args:
    input_dir    = "data/raw"                     # imagenes originales
    ndjson       = "data/raw/annotations.ndjson"  # anotaciones NDJSON (o None)
    output_dir   = "data/augmented"               # salida del dataset
    factor       = 12                             # imagenes generadas por original
    mosaic_ratio = 0.30                           # fraccion via mosaico
    seed         = 42
```

### 4.2 Verifica `data.yaml`

`data.yaml` apunta al dataset aumentado y define las clases. Debe coincidir con
el mapeo del NDJSON:

```yaml
path: .../data/augmented
train: images/train
val: images/val
test: images/test
nc: 5
names:
    0: metro
    1: destornillador
    2: martillo
    3: pinzas
    4: alicate
```

### 4.3 Opcion A — Todo de una (recomendado)

```bash
python run_pipeline.py --model_size n --epochs 100
```

El orquestador ejecuta: aumento → verifica `data.yaml` → entrena → evalua.
Si `data.yaml` ya existe, lo respeta; si no, lo genera desde `classes.txt`.

### 4.4 Opcion B — Modulo por modulo

```bash
# 1) Aumento + splits (usa la config interna del archivo)
python augmentation_pipeline.py

# 2) Entrenar y evaluar
python detector.py train    --data data.yaml --model_size n --epochs 100
python detector.py evaluate --data data.yaml --model models/best.pt

# (opcional) Predecir sobre nuevas imagenes
python detector.py predict --source data/raw --model models/best.pt
```

---

## 5. Como interpretar el mAP para la rubrica

- **mAP@0.5**: metrica principal (IoU ≥ 0.5). Objetivo **≥ 0.75**.
  `detector.py evaluate` imprime explicitamente `✅ PASSED` o `❌ FAILED`.
- **mAP@0.5:0.95**: metrica mas estricta (promedio IoU 0.5→0.95); suele ser menor.
- **Precision / Recall por clase**: identifican que herramientas fallan.
  - *Recall bajo* = no encuentra la herramienta (faltan ejemplos/aumento).
  - *Precision baja* = muchos falsos positivos (confunde clases).
- **Matriz de confusion** (`confusion_matrix.png`, en `runs/`): muestra
  confusiones entre clases parecidas (p.ej. `pinzas` vs `alicate`).

Para el informe: reporta mAP@0.5 global, la tabla por clase y la matriz de
confusion, y comenta las clases mas debiles y por que.

---

## 6. Troubleshooting: una clase tiene 0 detecciones o mAP muy bajo

Con anotacion manual, el problema casi siempre es de **datos**:

1. **Pocas instancias**: el Modulo 1 avisa si una clase queda con `<20`
   instancias en train. Sube `--factor` o agrega/anota mas imagenes de esa clase.
2. **Anotaciones inconsistentes**: revisa que el `class_id` en el NDJSON sea
   coherente y coincida con el orden de `data.yaml`.
3. **Desbalance severo**: si una clase domina, el split estratificado ayuda pero
   no lo arregla del todo; equilibra la cantidad de ejemplos por clase.
4. **Cajas mal ajustadas**: poligonos muy holgados generan bboxes gigantes tras
   `poligono_a_bbox`. Ajusta la anotacion.
5. **`min_visibility=0.4`** en el pipeline puede descartar objetos muy ocluidos
   tras un crop/rotacion; bajalo si pierdes muchas cajas.

---

## 7. Herramientas de anotacion manual

Cualquier herramienta que exporte NDJSON de Ultralytics (o YOLO `.txt`) sirve:

- **Ultralytics** (plataforma / HUB): exporta NDJSON directamente.
- **LabelImg** (`pip install labelImg`): exporta YOLO `.txt`. En ese caso pon
  `ndjson = None` en la config del Modulo 1 y deja los `.txt` junto a las
  imagenes en `input_dir`.
- **Label Studio** o **CVAT**: interfaz web, exportan a formatos YOLO.

Atajos utiles en LabelImg: `W` crea caja, `D` siguiente, `A` anterior, `Ctrl+S`
guarda. Recuerda: la calidad de la anotacion manual es el factor #1 para
alcanzar el mAP objetivo.

---

## Estructura del proyecto

```
.
├── augmentation_pipeline.py    # Modulo 1 (config interna en `class args`)
├── detector.py                 # Modulo 2 (CLI: train/predict/evaluate)
├── run_pipeline.py             # Modulo 3 (orquestador)
├── data.yaml                   # Config del dataset para YOLOv8
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                    # <- imagenes + annotations.ndjson (manual)
│   └── augmented/              # generado por el Modulo 1
│       ├── images/{train,val,test}
│       ├── labels/{train,val,test}
│       └── classes.txt
└── models/best.pt              # mejores pesos entrenados
```
