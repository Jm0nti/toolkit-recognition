# PROMPT-CONTEXTO PARA REDACTAR EL INFORME FINAL

Copia todo lo que sigue y pégalo en la IA que redactará el documento.

---

## TU ROL

Eres el redactor del **informe final escrito** de un proyecto universitario de Visión Artificial. Tu trabajo es convertir el contexto que te doy en un documento académico completo, claro y honesto.

Reglas de escritura obligatorias:
- **Español académico pero simple.** Frases cortas. Nada de relleno. Que no suene a IA.
- **Primera persona del plural**: "nosotros", "nuestro equipo", "decidimos", "probamos".
- **Explica el POR QUÉ de cada decisión técnica**, no solo el qué. La rúbrica premia la justificación.
- **Análisis crítico y honesto**: cuenta también lo que salió mal, las limitaciones y el domain shift. No exageres los resultados.
- **No inventes NADA.** Solo puedes usar las cifras y datos que están más abajo. Si te falta un dato para completar una frase, NO lo inventes: deja un marcador visible con el formato `[FALTA DATO: <qué falta>]` y sigue.
- Cada número que escribas debe venir del contexto. Si citas una métrica, cítala tal cual (mAP@0.5=0.956, etc.).

---

## CONTEXTO COMPLETO Y VERIFICADO DEL PROYECTO

### Identidad
- **Proyecto:** "Reconocimiento de herramientas manuales" — proyecto final de Visión Artificial, Grupo 3.
- **Objetivo:** detectar y reconocer herramientas manuales en fotos de cajas o superficies, con oclusión parcial, superposición y fondos variados.
- **10 clases:** metro, destornillador, martillo, pinzas, alicate, llave_inglesa, llave_combinada, brocha, espatula, llave_tubo.

### Dataset / Adquisición
- 4 colaboradores (jacob, juanl, juanma, tiago) tomaron fotos con su smartphone.
- ~30 imágenes originales → ~500 imágenes tras el aumento → ~3700 recortes (crops) para clasificación.
- Split 70/20/10 (train/val/test), estratificado.
- Anotación **manual** (polígonos de segmentación) en la plataforma Ultralytics, exportada como **NDJSON** por cada persona.
- Limitaciones de origen: pocas imágenes por clase, solo 4 personas, fondos e iluminación poco variados, sombras y reflejos metálicos.

### Pipeline (5 módulos)

**M1 — Unificación de anotaciones** (`src/data/merge_ndjson.py`): une los NDJSON de los 4 colaboradores. Remapea las clases **por NOMBRE, no por índice** (cada persona pudo etiquetar en distinto orden) hacia un orden canónico, y descarta clases fuera del set de 10.

**M2 — Aumento de datos** (`src/detection/augmentation_pipeline.py`): Albumentations con factor 12 (12 imágenes por original) + 30% de mosaicos. Transforms: RandomRotate90, HorizontalFlip, VerticalFlip, ShiftScaleRotate (rotate 45), RandomBrightnessContrast, HueSaturationValue, CLAHE, GaussNoise, MotionBlur, GaussianBlur, ImageCompression, CoarseDropout. Tamaño objetivo 640. Salida en formato YOLO. Por qué: teníamos muy pocas fotos, había que multiplicar y variar iluminación/ángulo/oclusión para que el modelo generalizara.

**M3 — Detección** (`src/detection/detector.py`): fine-tuning de **YOLOv8n (nano)**, con funciones train/predict/evaluate. Resultados reales (última época, split val, 100 épocas): **mAP@0.5=0.956, mAP@0.5:0.95=0.838, precisión=0.907, recall=0.898**. Meta académica mAP@0.5≥0.75 (superada con margen). Por qué nano: es ligero, entrena rápido y corre en CPU para la demo.

**M4 — Preprocesamiento + características + clasificación**
- Preprocesamiento clásico (`src/classification/preprocessing.py`): `to_grayscale`, `split_bgr` y `split_hsv` (separación de canales), `edges_canny` y `edges_sobel` (bordes), `resize_pad` (redimensión con relleno negro conservando aspecto).
- Características (`src/classification/features.py`):
  - `hog_features` = HOG sobre gris 128px, orientations=9, pixels_per_cell=16×16, cells_per_block=2×2, L2-Hys → captura **FORMA/bordes**.
  - `color_histogram` = histograma 3D HSV 8×8×8 normalizado → captura **COLOR**.
  - `combined_features` concatena ambos.
- `dataset_builder.py` recorta cada bbox del detector a una imagen por clase (los ~3700 crops).
- **3 modelos con la misma interfaz** (train/predict/save/load):
  - `hog_svm` = HOG + SVM (kernel RBF, StandardScaler). **accuracy=0.891, macro_f1=0.887. GANADOR.**
  - `color_hist_rf` = histograma HSV + RandomForest (200 árboles, class_weight balanced). accuracy=0.877, macro_f1=0.874.
  - `cnn` = CNN PyTorch ~200k params: 3 bloques Conv(32→64→128)-BN-ReLU-MaxPool, GlobalAvgPool, FC(128)-Dropout(0.3)-FC(clases). Entrada RGB 96×96. CrossEntropy ponderada por clase (compensa desbalance), early stopping paciencia 6, Adam lr=1e-3. accuracy=0.832, macro_f1=0.823.
- **N test = 495 recortes.** Tiempos: hog_svm train 15.1s / pred 6.0s; color_hist_rf 4.9s / 0.6s; cnn 368.4s / 1.3s.
- Peores clases por F1: hog_svm → alicate 0.83, espatula 0.85, pinzas 0.86; color_hist_rf → llave_tubo 0.82; cnn → llave_inglesa 0.63. Confusión típica: **pinzas ↔ alicate** (siluetas parecidas).

**M5 — Segmentación clásica** (`src/classification/segmentation.py`): `mask_otsu` (umbral global de Otsu + limpieza morfológica open/close), `mask_grabcut` (GrabCut inicializado con el bounding box del detector YOLO), `overlay_mask` (pinta la máscara y su contorno). Evidencia en `outputs/segmentacion/` (8 tiras: original | máscara Otsu | overlay Otsu | overlay GrabCut). **Hallazgo:** Otsu solo funciona bien con fondo uniforme y contrastado; en cajas con sombras o fondos oscuros la máscara global se degrada. GrabCut con el bbox recorta bien la silueta incluso en fondos difíciles, pero necesita esa inicialización. Por eso la segmentación "de producción" es la anotación poligonal + la detección aprendida, y Otsu/GrabCut quedan como comparación clásica.

### Frontend (la página)
FastAPI en `scripts/serve_app.py` + `web/index.html`. Endpoints: `POST /detectar` (sube imagen → corre YOLO → devuelve imagen anotada + JSON de detecciones), `GET /historial`, `GET /metricas`. Modelo en `runs/tools/weights/best.pt` con fallback a CPU. Pestañas: **Detector** (demo en vivo con slider de confianza e inventario) y **Métricas** (curvas y tablas). Diseño tipo taller (azul acero, cinta métrica amarilla).

### Pruebas reales de la demo (Playwright, hechas de verdad)
- En fotos tipo dataset detecta bien: p. ej. metro 0.96, destornilladores 0.94/0.93, martillo 0.94, pinzas 0.80, alicate 0.70.
- En fotos "de internet" se degrada (**domain shift**): confunde pinzas con alicate; tiene **sesgo de color** (aprendió que "amarillo compacto = metro": marcó una cinta PVC como metro 0.96 y NO reconoció un flexómetro verde); e ignora clases que no conoce (clavadora, gafas, escuadra). Esto es normal y esperado con ~500 imágenes de 4 personas.

### Limitaciones
Pocos datos por clase; poca variedad de fondos/personas; sin manejo de "desconocido"; sin NMS entre clases parecidas (de ahí los duplicados pinzas/alicate).

### Trabajo futuro
Más datos y más variados; balanceo de clases; transfer learning (MobileNetV2/EfficientNet); NMS agnóstico de clase; umbral de rechazo para lo desconocido.

---

## ESTRUCTURA DEL INFORME (SIGUE EXACTAMENTE ESTE ORDEN Y ESTOS PESOS)

El informe se ordena según la rúbrica. Escribe cada sección con su peso indicado como referencia de cuánta profundidad darle.

### 1. Introducción y planteamiento del problema
Qué explicar: qué son "herramientas manuales", por qué es un problema real y difícil (oclusión, superposición, metal que refleja), cuáles son las 10 clases, y el objetivo. Deja claro el reto desde el inicio.

### 2. Adquisición del dataset (8%)
Qué explicar: cómo y con qué se tomaron las fotos (4 personas, smartphone), cuántas originales, el split 70/20/10 estratificado, y cómo anotamos (polígonos en Ultralytics → NDJSON). Justifica: por qué anotación manual poligonal, por qué estratificar. Menciona el módulo M1 y **por qué remapeamos por nombre y no por índice**. Sé honesto con las limitaciones de origen.

> [IMAGEN: montaje con una foto real de cada uno de los 4 colaboradores]
> Pie de foto sugerido: "Muestra de las fotos originales tomadas por cada integrante del equipo con su smartphone."
> Insertar la imagen debajo de este párrafo. Archivo: usar la evidencia de `outputs/adquisicion/`.

### 3. Preprocesamiento (10%)
Qué explicar: el aumento de datos (M2) como paso central — factor 12, mosaicos 30%, la lista de transforms y **por qué** cada grupo (geométricas para ángulos, fotométricas para iluminación, ruido/blur para fotos de móvil, CoarseDropout para oclusión). Luego el preprocesamiento clásico de M4 (gris, canales BGR/HSV, bordes Canny/Sobel, resize_pad) y para qué sirve cada uno. Justifica el tamaño 640 y el formato YOLO.

> [IMAGEN: panel original | gris | canal H (HSV) | bordes Canny | bordes Sobel de un recorte real]
> Pie de foto sugerido: "Preprocesamiento de un recorte real: escala de grises, canal H del espacio HSV y detección de bordes con Canny y Sobel."
> Insertar la imagen debajo de este párrafo. Archivo: `outputs/preprocesamiento/`.

### 4. Segmentación (15%)
Qué explicar: los dos enfoques clásicos (Otsu con limpieza morfológica; GrabCut inicializado con el bbox de YOLO) y el overlay. Justifica: por qué probamos ambos, y el hallazgo clave (Otsu se degrada con sombras/fondo oscuro; GrabCut recorta mejor pero necesita el bbox). Cierra explicando **por qué la segmentación de producción es la anotación poligonal + la detección aprendida**, y Otsu/GrabCut quedan como comparación clásica. Esta sección pesa mucho (15%): dale desarrollo.

> [IMAGEN: tira comparativa original | máscara Otsu | overlay Otsu | overlay GrabCut]
> Pie de foto sugerido: "Comparación de segmentación clásica: Otsu falla con fondos con sombra, mientras GrabCut inicializado con el bounding box del detector recorta mejor la silueta."
> Insertar la imagen debajo de este párrafo. Archivo: una de las 8 tiras de `outputs/segmentacion/`.

### 5. Extracción de características (15%)
Qué explicar: HOG (forma/bordes) y el histograma HSV (color), con sus parámetros exactos, y la idea de `combined_features`. Justifica: **por qué HOG captura forma y el histograma captura color**, y por qué tiene sentido combinarlos para herramientas (donde la silueta importa más que el color). Menciona cómo `dataset_builder.py` genera los ~3700 crops desde las bboxes.

> [IMAGEN: visualización HOG (gradientes/silueta) + gráfico del histograma de color HSV]
> Pie de foto sugerido: "Características extraídas de un recorte: mapa de gradientes HOG (izquierda) que captura la forma, e histograma de color HSV (derecha) que captura el color."
> Insertar la imagen debajo de este párrafo. Archivo: `outputs/caracteristicas/`.

### 6. Clasificación (14%)
Qué explicar: los 3 modelos con la misma interfaz (hog_svm, color_hist_rf, cnn), su diseño y sus resultados. Presenta las métricas en **tabla** (ver formato abajo). Justifica: **por qué ganó hog_svm** (la forma discrimina mejor que el color en herramientas), por qué la CNN quedó por debajo (pocos datos), y comenta los tiempos (SVM y RF entrenan en segundos, la CNN en minutos). Analiza las peores clases por F1 y la confusión **pinzas ↔ alicate** por siluetas parecidas.

> [IMAGEN: matrices de confusión de los 3 modelos + gráfico comparativo]
> Pie de foto sugerido: "Matrices de confusión de los tres clasificadores y comparación de accuracy/macro-F1. El modelo HOG+SVM obtiene el mejor desempeño."
> Insertar la imagen debajo de este párrafo. Archivos: `outputs/ml_reports/`.

### 7. Reconocimiento de patrones, detección y métricas (8%)
Qué explicar: la detección con YOLOv8n (M3) como el corazón del reconocimiento, sus resultados reales (mAP@0.5=0.956, mAP@0.5:0.95=0.838, precisión=0.907, recall=0.898, 100 épocas) y la comparación con la meta (≥0.75). Justifica la elección de nano. Presenta las métricas en **tabla**. Menciona la calidad de código y la organización en GitHub (pipeline modular M1–M5).

> [IMAGEN: curvas PR / P / R / F1, matriz de confusión del detector y lotes de validación anotados]
> Pie de foto sugerido: "Evaluación del detector YOLOv8n: curvas de precisión-recall y ejemplos del conjunto de validación con las detecciones."
> Insertar la imagen debajo de este párrafo. Archivos: `outputs/metricas/`.

### 8. Aplicación (frontend y demo en vivo)
Qué explicar: la página FastAPI, los endpoints, las dos pestañas, el modelo con fallback a CPU. Cuenta las **pruebas reales de la demo**: funciona bien en fotos tipo dataset (da los ejemplos con sus confianzas) y se **degrada en fotos de internet** (domain shift, sesgo de color del metro amarillo, clases desconocidas ignoradas). Explica **por qué pasa esto** (dataset pequeño y poco variado) con honestidad.

> [IMAGEN: captura de la pestaña Detector con una detección real y su inventario]
> Pie de foto sugerido: "Demo en vivo: el usuario sube una foto y la aplicación devuelve la imagen anotada más el inventario de herramientas detectadas."
> Insertar la imagen debajo de este párrafo. Archivo: captura de la aplicación (si no existe, dejar `[FALTA DATO: captura de la demo]`).

### 9. Conclusiones, limitaciones y trabajo futuro (parte de Presentación, 30%)
Qué explicar: qué logramos (cumplimos y superamos la meta), qué aprendimos, las limitaciones (pocos datos, 4 personas, sin manejo de "desconocido", sin NMS entre clases parecidas), y el trabajo futuro (más datos, balanceo, transfer learning, NMS agnóstico de clase, umbral de rechazo). Cierra con una reflexión crítica honesta.

---

## FORMATO DE LAS TABLAS DE MÉTRICAS

Usa tablas Markdown. Modelo para clasificación:

| Modelo | Accuracy | Macro-F1 | Tiempo train | Tiempo pred |
|---|---|---|---|---|
| HOG + SVM | 0.891 | 0.887 | 15.1s | 6.0s |
| Color HSV + RandomForest | 0.877 | 0.874 | 4.9s | 0.6s |
| CNN | 0.832 | 0.823 | 368.4s | 1.3s |

(N test = 495 recortes.)

Modelo para detección:

| Métrica | Valor | Meta académica |
|---|---|---|
| mAP@0.5 | 0.956 | ≥0.75 |
| mAP@0.5:0.95 | 0.838 | — |
| Precisión | 0.907 | — |
| Recall | 0.898 | — |

---

## INSTRUCCIONES FINALES DE FORMATO
- Títulos con jerarquía (`#`, `##`, `###`) siguiendo las 9 secciones de arriba.
- Todas las métricas en tablas; nunca sueltas en medio de un párrafo largo.
- Cada imagen va con su marcador `[IMAGEN: ...]`, su pie de foto y la frase "insertar la imagen debajo de este párrafo".
- Extensión sugerida: **6 a 9 páginas** (sin contar imágenes).
- Recuerda: cero datos inventados. Todo dato ausente va como `[FALTA DATO: ...]`.

## LISTA DE VERIFICACIÓN FINAL (revísala antes de entregar)
- [ ] ¿Están las 9 secciones en el orden de la rúbrica?
- [ ] ¿Cada sección justifica el POR QUÉ de sus decisiones?
- [ ] ¿Cada sección tiene su marcador `[IMAGEN: ...]` con pie de foto y carpeta de `outputs/` indicada?
- [ ] ¿Todas las métricas coinciden EXACTAMENTE con el contexto (mAP, accuracy, F1, tiempos)?
- [ ] ¿No hay ningún dato inventado? ¿Los faltantes están como `[FALTA DATO: ...]`?
- [ ] ¿Se cuenta con honestidad el domain shift y las limitaciones?
- [ ] ¿Está todo en primera persona del plural, en español simple y sin relleno?
- [ ] ¿Adquisición(8%), Preprocesamiento(10%), Segmentación(15%), Características(15%), Clasificación(14%), Reconocimiento/métricas(8%) están todos cubiertos?

Ahora redacta el informe completo.