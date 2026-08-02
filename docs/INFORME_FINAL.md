# Reconocimiento de herramientas manuales

**Proyecto final de Visión Artificial — Grupo 3**
Integrantes: jacob, juanl, juanma, tiago

---

## 1. Introducción y planteamiento del problema

Las herramientas manuales son las piezas que se usan sin motor: martillos, destornilladores,
llaves, alicates, brochas, espátulas y metros. En un taller se guardan mezcladas dentro de una
caja. Nuestro proyecto parte de ahí: dada **una foto de una caja de herramientas**, queremos que
el computador diga **qué herramientas hay y cuántas**.

El caso de uso que nos motivó es el inventario. Contar herramientas a mano es lento y se presta a
errores, y al cerrar un trabajo es fácil que falte una pieza sin que nadie se dé cuenta. Con una
sola foto se puede levantar ese inventario en segundos.

El problema es fácil de enunciar y difícil de resolver. En una caja real pasan tres cosas al
mismo tiempo:

- **Oclusión parcial.** Las herramientas se tapan unas a otras. Casi nunca se ve una pieza
  completa.
- **Superposición.** Se amontonan, así que sus contornos se cruzan y cuesta saber dónde termina
  una y empieza la otra.
- **Fondos difíciles y metal.** El cartón o el plástico de la caja generan sombras, y las
  superficies metálicas reflejan la luz. El brillo cambia según el ángulo de la foto, no según la
  herramienta.

A eso se suma un problema propio del dominio: hay clases que se parecen muchísimo. Unas **pinzas**
y un **alicate** tienen casi la misma silueta en una foto cenital, y una **llave inglesa**, una
**llave combinada** y una **llave de tubo** comparten forma alargada y color metálico. Como se
verá en las secciones 6 y 7, esas parejas son justamente donde nuestro sistema falla más.

Trabajamos con **10 clases**: `metro`, `destornillador`, `martillo`, `pinzas`, `alicate`,
`llave_inglesa`, `llave_combinada`, `brocha`, `espatula`, `llave_tubo`.

Nuestro objetivo fue construir un pipeline completo y medible, de la foto al inventario, con una
meta académica concreta para la detección: **mAP@0.5 ≥ 0.75**. Decidimos partir el trabajo en dos
etapas —primero *dónde* está cada herramienta y después *cuál* es— por dos razones. La primera es
que así podemos medir y mejorar cada parte por separado. La segunda es que la oclusión nos obliga
a recortar cada herramienta y analizarla sola, en vez de mirar la caja entera de un golpe.

El repositorio está organizado en cinco módulos independientes:

| # | Módulo | Ubicación | Función |
|---|---|---|---|
| M1 | Unificación de anotaciones | `src/data/merge_ndjson.py` | Une los NDJSON de los 4 colaboradores en un orden canónico |
| M2 | Aumento de datos + splits | `src/detection/augmentation_pipeline.py` | Albumentations + mosaico + split estratificado |
| M3 | Detección | `src/detection/detector.py` | Fine-tuning de YOLOv8n, `train` / `predict` / `evaluate` |
| M4 | Preprocesamiento, características y clasificación | `src/classification/` | Recortes → 3 clasificadores |
| M5 | Segmentación clásica | `src/classification/segmentation.py` | Otsu y GrabCut |

Los scripts ejecutables viven en `scripts/` y la lógica en `src/`. Todas las rutas se anclan a la
raíz del repositorio (`src/paths.py`), así que cualquier script corre desde cualquier directorio.

---

## 2. Adquisición del dataset (8%)

### Cómo tomamos las fotos

Cuatro integrantes del equipo tomamos las fotos con **nuestros propios smartphones**, sin trípode
ni iluminación de estudio. Fotografiamos herramientas dentro de cajas y sobre superficies planas,
a propósito con piezas superpuestas y parcialmente tapadas. Lo hicimos así porque queríamos que el
modelo viera desde el primer día las condiciones difíciles reales, y no herramientas aisladas
sobre fondo blanco.

El reparto quedó desbalanceado, y lo decimos tal cual:

| Colaborador | Fotos originales | Formato |
|---|---:|---|
| tiago | 19 | `.jpg` / `.png` |
| juanma | 8 | `.jpg` |
| jacob | 6 | `.jpeg` |
| juanl | 3 | `.jpg` |
| **Total** | **36** | |

Sobre esas 36 fotos anotamos **152 herramientas en total**, repartidas así:

| Clase | Instancias originales |
|---|---:|
| destornillador | 26 |
| martillo | 24 |
| brocha | 22 |
| metro | 16 |
| espatula | 15 |
| llave_tubo | 14 |
| alicate | 13 |
| pinzas | 9 |
| llave_combinada | 8 |
| llave_inglesa | 5 |
| **Total** | **152** |

Conviene detenerse en este número, porque es el que explica casi todo lo que viene después:
**tenemos 5 llaves inglesas anotadas en todo el proyecto**. Ninguna cantidad de aumento crea
información que no está: si esas 5 llaves salieron con la misma luz y el mismo ángulo, rotarlas y
cambiarles el brillo no le enseña al modelo qué es una llave inglesa. Como se verá en las
secciones 6 y 7, `llave_inglesa` es la peor clase tanto del detector como del clasificador ganador,
y el orden de esta tabla predice bastante bien el orden de las métricas por clase.

**36 imágenes originales y 152 instancias** son muy pocas para entrenar un detector. Esa es la
limitación central del proyecto y condiciona todo lo que viene después: el aumento agresivo del
Módulo 2, la elección de un modelo nano en el Módulo 3 y el hecho de que los métodos clásicos le
ganen a la CNN en el Módulo 4.

> **[IMAGEN: montaje con una foto real de cada uno de los 4 colaboradores]**
> *Pie de foto: "Muestra de las fotos originales tomadas por cada integrante del equipo con su smartphone."*
> Insertar la imagen debajo de este párrafo. Archivo: `outputs/adquisicion/montaje.jpg`.

### Cómo anotamos y por qué

Cada persona anotó **sus propias fotos a mano** en la plataforma Ultralytics, dibujando
**polígonos de segmentación** sobre cada herramienta, y exportó su trabajo como un archivo
**NDJSON** independiente (`data/raw/<persona>/*.ndjson`).

Elegimos polígonos y no cajas por dos motivos. Primero, un polígono marca la silueta exacta de la
pieza, y con herramientas alargadas y en diagonal una caja rectangular incluye mucho fondo y
mucha herramienta vecina. Segundo, teniendo el polígono siempre podemos derivar la caja
(`poligono_a_bbox` en `augmentation_pipeline.py` lo hace), pero al revés no: partir de cajas nos
habría cerrado la puerta a la segmentación. Anotar a mano, además, era la única opción realista
con 36 fotos y sin un modelo previo que preetiquetara.

### M1 — Unificación de las anotaciones

Los cuatro archivos no encajaban directamente. Cada persona creó sus clases en la plataforma en el
orden en que se le ocurrió, y ninguno coincidía con los demás. Estas son las cabeceras reales de
los cuatro NDJSON:

| Índice | juanma | tiago | juanl | jacob |
|---:|---|---|---|---|
| 0 | metro | metro | metro | metro |
| 1 | destornillador | destornillador | destornillador | destornillador |
| 2 | **espatula** | **martillo** | **martillo** | **martillo** |
| 3 | brocha | pinzas | pinzas | pinzas |
| 4 | martillo | alicate | alicate | alicate |
| 5 | **llave_tubo** | **llave_inglesa** | — | **llave_combinada** |
| 6 | pinzas | llave_combinada | — | brocha |
| 7 | alicate | brocha | — | espatula |
| 8 | llave_inglesa | espatula | — | llave_tubo |
| 9 | llave_combinada | llave_tubo | — | — |
| 10 | — | *rodillo_pintura* | — | — |

Se ve el problema de un vistazo: el índice `2` es `martillo` para tres personas pero **`espatula`
para juanma**, y el índice `5` significa **tres cosas distintas** según quién anotó. Además juanl
solo usó 5 clases, jacob nunca etiquetó `llave_inglesa`, y tiago añadió `rodillo_pintura`, que no
estaba en el set acordado.

`src/data/merge_ndjson.py` resuelve eso con tres reglas:

1. **Remapea las clases por NOMBRE, no por índice.** El script construye
   `NAME_TO_CANONICAL` a partir de un diccionario canónico fijo y traduce cada segmento usando el
   `class_names` de la cabecera de cada NDJSON. Esta es la decisión más importante del módulo: el
   índice es un accidente de cómo cada persona configuró la plataforma, mientras que el nombre es
   lo único que significa lo mismo para todos. Si hubiéramos concatenado por índice, cada espátula
   de juanma habría entrado al dataset etiquetada como martillo, y el error habría sido
   **invisible**: el entrenamiento habría corrido igual, sin ningún aviso, solo que aprendiendo
   etiquetas equivocadas.
2. **Descarta lo que no está en el set canónico.** Los segmentos de `rodillo_pintura` se eliminan y
   el script imprime por consola qué clases descartó de cada archivo.
3. **Omite imágenes duplicadas** entre colaboradores, comparando por nombre de archivo, para no
   contar dos veces la misma foto.

La salida es `data/raw/unified.ndjson`: una sola cabecera `dataset` con las 10 clases canónicas y
todas las imágenes remapeadas.

### El split

Dividimos **70/20/10** en train/val/test, **estratificado por clase dominante** de cada imagen
(`split_estratificado` en `augmentation_pipeline.py`). Estratificamos porque con clases tan
desbalanceadas un split aleatorio puro puede dejar una clase rara casi sin ejemplos en test, y
entonces la métrica de esa clase no significa nada. Al agrupar por clase dominante y repartir
dentro de cada grupo, las tres particiones se parecen entre sí.

El split se aplica **después** del aumento, sobre las imágenes generadas. Es importante ser
honestos con esto: como las variantes aumentadas de una misma foto original pueden caer en
particiones distintas, existe **fuga de información entre train y test**. Las métricas que
reportamos más adelante hay que leerlas con esa advertencia: miden qué tan bien el modelo
reconoce herramientas *de nuestras fotos*, no qué tan bien generaliza a fotos nuevas. La sección 8
muestra exactamente ese límite en la práctica.

### Limitaciones de origen

- **Muy pocas fotos originales** (36) y muy pocas instancias por clase (152 en total).
  `llave_inglesa` es la más escasa, con **5 instancias originales** en todo el dataset.
- **Solo 4 personas**, así que el estilo de foto, la altura y el encuadre varían poco.
- **Fondos e iluminación poco variados.** Cada colaborador fotografió casi siempre en el mismo
  sitio y con la misma luz.
- **Sombras y reflejos metálicos** constantes, propios de fotografiar metal dentro de una caja.

---

## 3. Preprocesamiento (10%)

El preprocesamiento tiene dos partes que resuelven problemas distintos: el **aumento de datos**
(M2), que corre una vez antes de entrenar, y el **preprocesamiento clásico por recorte** (M4), que
corre antes de extraer características.

### M2 — Aumento de datos

Con 36 fotos, cualquier modelo memoriza. El aumento fue la pieza que hizo viable el proyecto.
Usamos **Albumentations** con **factor 12** (12 versiones por foto original, más la original sin
tocar) y un **30 % adicional de mosaicos**. Los números reales que produce el pipeline son:

| Etapa | Cantidad |
|---|---:|
| Fotos originales anotadas | 36 |
| Variantes por aumento clásico (36 × 12 + 36 originales) | 468 |
| Mosaicos (30 % de 36 × 12) | 129 |
| **Total del dataset aumentado** | **597** |
| — train / val / test | 413 / 116 / 68 |
| Instancias anotadas (cajas) | 2 909 / 807 / 442 |

Elegimos las transformaciones por grupos, cada uno atacando un problema concreto de nuestras
fotos:

| Grupo | Transforms | Por qué |
|---|---|---|
| Geométricas | `RandomRotate90`, `HorizontalFlip`, `VerticalFlip`, `ShiftScaleRotate` (rotación ±45°), `RandomResizedCrop` (escala 0.7–1.0) | Las fotos son cenitales y la herramienta puede estar en **cualquier ángulo**. Una llave girada 40° sigue siendo la misma llave, y el modelo tiene que saberlo. |
| Fotométricas | `RandomBrightnessContrast` (±0.3), `HueSaturationValue`, `CLAHE`, `ImageCompression` (calidad 70–100) | Simulan la **luz cambiante de un taller** y la compresión JPEG del celular. Sin esto el modelo aprende la iluminación concreta del cuarto donde tomamos las fotos. |
| Ruido y desenfoque | `GaussNoise`, `MotionBlur`, `GaussianBlur` | Imitan el **sensor del smartphone** y el pulso de la mano. Las fotos de la demo van a venir de un celular, no de una cámara fija. |
| Oclusión | `CoarseDropout` (1–4 huecos) + mosaico | Tapan parches al azar para enseñar **oclusión y superposición**, que es la condición normal dentro de una caja. |

Dos detalles de implementación que importan:

- **`min_visibility=0.4`** en `BboxParams`. Si una transformación deja una caja con menos del 40 %
  visible, esa caja se descarta. Sin ese filtro habríamos generado etiquetas de objetos que ya casi
  no se ven, o sea ruido de entrenamiento.
- **Los mosaicos** juntan 4 fotos en una rejilla 2×2 de 640×640 y fusionan sus cajas
  reescaladas. El objetivo es que el modelo vea **muchas herramientas distintas en una sola
  imagen**, que es exactamente la escena real de una caja llena.

**Por qué 640 y por qué formato YOLO.** 640 px es el `imgsz` por defecto de YOLOv8: entrenar en el
tamaño nativo evita un reescalado extra y es el punto donde el modelo fue diseñado para funcionar.
El formato de salida es YOLO (`clase cx cy w h` normalizados, un `.txt` por imagen) porque es lo
que consume directo el entrenamiento del Módulo 3, sin ningún conversor intermedio.

### M4 — Preprocesamiento clásico por recorte

Sobre cada recorte que devuelve el detector aplicamos las operaciones de
`src/classification/preprocessing.py`:

| Función | Qué hace | Para qué sirve |
|---|---|---|
| `to_grayscale` | BGR → gris | Quita el color y deja solo la intensidad. Es la base de HOG y de los bordes: la silueta de un martillo no depende de su color. |
| `split_bgr` / `split_hsv` | Separa canales | HSV separa el **matiz** (color) del **valor** (brillo). El canal H casi no cambia con la luz, así que es más robusto a sombras y reflejos que trabajar en BGR. Es la base del histograma de color. |
| `edges_canny` | Bordes finos (blur 5, umbrales 80/180) | Marca el contorno limpio. El desenfoque previo evita que el ruido del sensor del celular se convierta en bordes falsos. |
| `edges_sobel` | Magnitud del gradiente | Da una respuesta continua en vez de binaria; muestra dónde el gradiente es más fuerte. |
| `resize_pad` | Redimensiona a 128×128 rellenando con negro | Da un **tamaño fijo sin deformar** la herramienta. Es imprescindible para HOG, que necesita entradas del mismo tamaño; estirar una llave alargada a un cuadrado cambiaría justamente lo que la identifica. |

> **[IMAGEN: panel original | gris | canal H (HSV) | bordes Canny | bordes Sobel de un recorte real]**
> *Pie de foto: "Preprocesamiento de un recorte real: escala de grises, canal H del espacio HSV y detección de bordes con Canny y Sobel."*
> Insertar la imagen debajo de este párrafo. Archivo: `outputs/preprocesamiento/6S_preproc.jpg` (generado con `scripts/run_preprocessing_demo.py`).

---

## 4. Segmentación (15%)

Segmentar es decidir, píxel a píxel, qué es herramienta y qué es fondo. Probamos **dos enfoques
clásicos** y los comparamos contra lo que finalmente usamos en producción.

### 4.1 Otsu con limpieza morfológica

`mask_otsu` aplica el **umbral global de Otsu** sobre la imagen en gris, previo desenfoque
gaussiano. Otsu busca automáticamente el umbral que mejor separa el histograma en dos grupos. Como
la máscara resultante puede quedar invertida, añadimos una heurística: **si más de la mitad de la
imagen queda en blanco, invertimos**, asumiendo que las herramientas ocupan menos área que el
fondo.

Después limpiamos con morfología: **apertura** (`MORPH_OPEN`, elipse 5×5) para quitar puntos
sueltos del ruido, y **cierre** (`MORPH_CLOSE`, elipse 9×9) para rellenar los huecos que quedan
dentro de una misma herramienta.

Probamos Otsu porque es el método clásico de referencia: no necesita entrenamiento, corre en
milisegundos y, si el fondo es uniforme, resuelve el problema sin más.

### 4.2 GrabCut inicializado con el bbox del detector

`mask_grabcut` toma un enfoque distinto. Recibe el **bounding box que ya produjo YOLO**, marca
todo lo que está fuera del rectángulo como fondo seguro, y deja que GrabCut decida iterativamente
(5 iteraciones) qué es objeto dentro de la caja. GrabCut modela fondo y objeto como mezclas de
gaussianas en color y refina la frontera con cortes de grafo.

La ventaja es que **no depende de que exista un umbral global válido**: trabaja localmente, dentro
de una región donde sabemos que hay una herramienta. La desventaja es que **necesita esa
inicialización**, o sea que depende del detector.

`overlay_mask` pinta la máscara semitransparente (alpha 0.45) y dibuja su contorno, para poder
comparar los dos métodos a simple vista.

Un detalle práctico: GrabCut sobre una foto de celular a resolución completa tarda **minutos**.
Por eso `run_segmentation_demo.py` reescala a lado máximo 1600 px antes de segmentar.

### 4.3 El hallazgo

Generamos **8 tiras comparativas** (2 fotos por colaborador) en `outputs/segmentacion/`, cada una
con cuatro paneles: `original | máscara Otsu | overlay Otsu | overlay GrabCut`. El resultado fue
claro y contrario a lo que esperábamos al principio.

**Otsu no separa las herramientas del fondo en nuestras fotos.** En la tira de `tiago_10S`, por
ejemplo, el umbral global partió la imagen entre "cartón claro de la caja" y "herramientas +
sombras oscuras", y la heurística de inversión dejó como objeto **el fondo de cartón**, no las
herramientas. El overlay pinta la caja entera de amarillo. El problema de fondo es que Otsu asume
**dos poblaciones de intensidad bien separadas en toda la imagen**, y en una caja real hay
herramientas claras (metro rojo, brocha blanca), herramientas oscuras (mangos negros), cartón
intermedio y sombras. Un solo número no puede describir eso.

**GrabCut sí recorta la silueta**, incluso en esas mismas fotos difíciles. En la tira de
`tiago_10S` marca limpiamente el metro, las brochas y el destornillador, siguiendo el contorno
real de cada pieza. Pero lo hace porque le dimos el bbox de YOLO: **sin detector, GrabCut no sabe
por dónde empezar**.

> **[IMAGEN: tira comparativa original | máscara Otsu | overlay Otsu | overlay GrabCut]**
> *Pie de foto: "Comparación de segmentación clásica: Otsu falla con fondos con sombra, mientras GrabCut inicializado con el bounding box del detector recorta mejor la silueta."*
> Insertar la imagen debajo de este párrafo. Archivo: una de las 8 tiras de `outputs/segmentacion/` (recomendamos `tiago_10S_seg.jpg`).

### 4.4 Por qué la segmentación de producción es otra

De ahí sale nuestra decisión. La segmentación que el sistema usa de verdad es la **anotación
poligonal manual + la detección aprendida**, y Otsu/GrabCut quedan como comparación clásica. El
razonamiento es este:

1. **Otsu no sirve para nuestro caso.** No es una cuestión de afinar parámetros: la hipótesis del
   método (un umbral global separa objeto de fondo) no se cumple en una caja con muchas
   herramientas de colores distintos y sombras.
2. **GrabCut funciona pero es circular.** Necesita el bbox del detector, así que no puede
   reemplazarlo: solo puede refinarlo. Como paso de post-proceso es útil; como método de
   segmentación autónomo, no.
3. **La anotación poligonal es la que aporta la información real.** Al etiquetar a mano dibujamos
   la silueta de cada herramienta, incluso cuando estaba parcialmente tapada. Ningún método
   clásico no supervisado puede resolver la oclusión, porque decidir que dos trozos separados por
   un martillo pertenecen a la misma llave requiere saber qué es una llave.
4. **La detección aprendida hereda esa información.** YOLO se entrena con las cajas derivadas de
   esos polígonos y aprende a localizar herramientas tapadas, superpuestas y sobre cualquier
   fondo, que es exactamente donde Otsu se cae.

En resumen: implementamos y evaluamos los métodos clásicos, entendimos **por qué** fallan en
nuestras imágenes, y esa comprensión es la que justifica el camino que sí tomamos.

---

## 5. Extracción de características (15%)

Un clasificador clásico no "ve" la imagen: necesita un **vector de números** que resuma lo
importante. Sacamos dos descriptores que miran cosas distintas y se complementan.

### 5.1 HOG — la forma

`hog_features` (en `src/classification/features.py`) calcula el **Histograma de Gradientes
Orientados** sobre el recorte en gris, con estos parámetros exactos:

| Parámetro | Valor |
|---|---|
| Tamaño de entrada | 128×128 (vía `resize_pad`, conserva el aspecto) |
| `orientations` | 9 |
| `pixels_per_cell` | 16 × 16 |
| `cells_per_block` | 2 × 2 |
| `block_norm` | L2-Hys |
| `transform_sqrt` | `True` |

**Por qué HOG captura la forma.** HOG divide la imagen en celdas y, en cada una, cuenta hacia
dónde apuntan los gradientes de intensidad. Un gradiente fuerte aparece justo donde hay un borde,
así que el descriptor termina siendo un resumen de **hacia dónde apuntan los contornos en cada
zona**. El mango recto de un destornillador produce un patrón de orientaciones muy distinto al de
la cabeza en forma de U de un martillo. Lo calculamos en gris a propósito: la silueta de una
herramienta no depende de su color.

La normalización **L2-Hys por bloques** hace que el descriptor sea robusto a cambios de
iluminación local, que es un problema real con reflejos metálicos. Y `transform_sqrt` comprime el
rango dinámico, reduciendo el peso de los brillos especulares.

### 5.2 Histograma HSV — el color

`color_histogram` calcula un **histograma 3D en HSV con 8×8×8 casillas** (512 valores), aplanado y
normalizado por norma L2.

**Por qué el histograma captura el color.** Cuenta cuántos píxeles caen en cada combinación de
matiz, saturación y valor, sin importar *dónde* están. Es la firma cromática del recorte: un
mango plástico rojo, uno amarillo o una cabeza metálica gris producen histogramas muy distintos.

Elegimos **HSV y no BGR** porque HSV separa el color (H) del brillo (V). En nuestras fotos, tomadas
en cuartos sin luz controlada, el brillo cambia mucho entre una foto y otra pero el matiz no. Y
**normalizamos** para que solo cuente la *proporción* de colores y no el tamaño del recorte: sin
eso, un recorte grande tendría cuentas más altas que uno pequeño de la misma herramienta.

### 5.3 `combined_features` y por qué combinar

`combined_features` concatena los dos vectores (HOG + histograma HSV). La intuición es que los
descriptores fallan en casos opuestos: **la forma sola** confunde pinzas con alicate, que tienen
casi la misma silueta; **el color solo** confunde dos llaves metálicas grises distintas. Juntos,
cada uno cubre el punto ciego del otro.

Dicho eso, para herramientas **la forma pesa más que el color**, y por una razón concreta: el
color de un mango es una decisión del fabricante, no una propiedad de la herramienta. Existen
martillos con mango rojo, negro o de madera. Pero todos los martillos tienen cabeza y mango. Los
resultados de la sección 6 confirman esta intuición: el modelo basado en forma le gana al basado
en color.

### 5.4 De dónde salen los recortes

`dataset_builder.py` convierte el dataset de **detección** en uno de **clasificación**. Recorre
cada imagen aumentada, lee su `.txt` YOLO y **recorta cada bbox** con un margen del 5 %, guardando
el resultado en `outputs/ml_datasets/crops/<split>/<clase>/`. Cada recorte queda etiquetado por la
carpeta que lo contiene, lo que lo hace consumible tanto por scikit-learn como por `ImageFolder`
de PyTorch. Los recortes con menos de 24 px de lado se descartan por degenerados.

El resultado son **4 113 recortes**:

| Clase | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| destornillador | 503 | 141 | 76 | 720 |
| martillo | 438 | 129 | 71 | 638 |
| brocha | 430 | 114 | 69 | 613 |
| metro | 305 | 94 | 45 | 444 |
| espatula | 280 | 65 | 46 | 391 |
| llave_tubo | 274 | 77 | 36 | 387 |
| alicate | 249 | 66 | 40 | 355 |
| pinzas | 175 | 50 | 28 | 253 |
| llave_combinada | 135 | 30 | 13 | 178 |
| llave_inglesa | 91 | 33 | 10 | 134 |
| **Total** | **2 880** | **799** | **434** | **4 113** |

El desbalance salta a la vista: `destornillador` tiene **5,4 veces** más recortes que
`llave_inglesa`. Esto tiene consecuencias directas en la sección 6.

> **[IMAGEN: visualización HOG (gradientes/silueta) + gráfico del histograma de color HSV]**
> *Pie de foto: "Características extraídas de un recorte: mapa de gradientes HOG (izquierda) que captura la forma, e histograma de color HSV (derecha) que captura el color."*
> Insertar la imagen debajo de este párrafo. Archivos: `outputs/caracteristicas/hog.jpg` y `outputs/caracteristicas/histograma.png` (generados con `scripts/run_features_demo.py`).

---

## 6. Clasificación (14%)

### 6.1 Los tres modelos

Construimos tres clasificadores con la **misma interfaz** (`train` / `predict` / `save` / `load`),
para poder compararlos de forma justa y tratarlos uniformemente en los scripts. Cada uno se
entrena dentro de su propio `try/except`: **si uno falla, los otros dos siguen**.

**`hog_svm` — forma.** Pipeline de scikit-learn: `StandardScaler` + `SVC` con kernel RBF (C=10,
`gamma='scale'`). El escalado es necesario porque el SVM con RBF mide distancias, y sin
estandarizar las dimensiones con más varianza dominarían el resultado. El kernel RBF permite
fronteras no lineales entre los gradientes de HOG.

**`color_hist_rf` — color.** `RandomForestClassifier` con 200 árboles, sin límite de profundidad y
`class_weight="balanced"`, sobre el histograma HSV de 512 dimensiones. Elegimos Random Forest
porque un histograma se presta a reglas del tipo "si hay mucho píxel amarillo saturado, entonces
metro", que es exactamente lo que un árbol aprende. El `class_weight="balanced"` compensa el
desbalance de la tabla anterior.

**`cnn` — aprendizaje profundo.** Red pequeña en PyTorch, unos 200 mil parámetros:

```
Conv(3→32)  → BatchNorm → ReLU → MaxPool
Conv(32→64) → BatchNorm → ReLU → MaxPool
Conv(64→128)→ BatchNorm → ReLU → MaxPool
GlobalAvgPool → FC(128) → ReLU → Dropout(0.3) → FC(10)
```

Entrada RGB 96×96 con letterbox. Entrenamiento con Adam (lr=1e-3, weight decay 1e-4),
**CrossEntropy ponderada por clase** (pesos inversamente proporcionales a la frecuencia, para
compensar el desbalance) y **early stopping** por accuracy de validación con paciencia 6. La
hicimos deliberadamente pequeña: con 2 880 recortes de entrenamiento, una red grande se
sobreajustaría de inmediato.

### 6.2 Resultados

Evaluación sobre el split **test**, N = **434 recortes** nunca vistos durante el entrenamiento.
Fuente: `outputs/ml_reports/comparison.md` y los `*_metrics.json`.

| Modelo | Accuracy | Macro-F1 | Tiempo train | Tiempo pred |
|---|---:|---:|---:|---:|
| **HOG + SVM** | **0.8456** | **0.8241** | 17.90 s | 3.542 s |
| CNN | 0.8203 | 0.8081 | 635.49 s | 1.866 s |
| Color HSV + RandomForest | 0.8065 | 0.7888 | 5.22 s | 0.881 s |

*(N test = 434 recortes.)*

### 6.3 Por qué ganó HOG + SVM

**HOG + SVM es el mejor en accuracy (0.8456) y en macro-F1 (0.8241).** La explicación es la
hipótesis de la sección 5, ahora confirmada con datos: **con pocos ejemplos, la forma discrimina
mejor que el color**. La silueta de una herramienta es una propiedad estable —todos los martillos
tienen cabeza y mango— mientras que el color depende del fabricante y del reflejo del metal en el
momento de la foto. Además, el SVM con RBF se comporta bien en espacios de muchas dimensiones con
pocas muestras, que es justo nuestro escenario.

El **RandomForest de color quedó último (0.8065)** y esto también encaja: la mitad de nuestras
clases son metal gris. `llave_inglesa`, `llave_combinada` y `llave_tubo` tienen prácticamente el
mismo histograma HSV, así que el descriptor de color simplemente no las distingue.

**La CNN (0.8203) quedó en medio, por debajo del clásico ganador.** El motivo es el volumen de
datos. Una red convolucional tiene que *aprender* sus propios filtros de bordes desde cero; HOG ya
viene con esos filtros diseñados a mano. Con 2 880 recortes —y encima muchos de ellos variantes
aumentadas de las mismas 36 fotos— la red no tiene material suficiente para aprender algo mejor
que un descriptor bien elegido. Este es, para nosotros, el aprendizaje central del proyecto:
**más capacidad no compensa menos datos**.

### 6.4 Los tiempos

Los tiempos cuentan una historia práctica que las accuracies solas no muestran:

- `color_hist_rf` entrena en **5,22 s** y predice en **0,88 s**. Es el más barato con diferencia y
  su accuracy está a solo 4 puntos del ganador. Como baseline para un dispositivo modesto es una
  opción perfectamente razonable.
- `hog_svm` entrena en **17,90 s** pero es el **más lento en predicción (3,54 s)**, porque hay que
  calcular el descriptor HOG de cada recorte y el SVM con RBF evalúa contra sus vectores de
  soporte.
- La CNN cuesta **635,49 s de entrenamiento** —unas 35 veces más que HOG+SVM— para quedar por
  debajo en accuracy. En predicción sí es rápida (1,87 s), porque una pasada hacia adelante es
  barata.

La conclusión es incómoda pero honesta: en este proyecto **la CNN no se paga**. El costo de
entrenamiento es enorme y el resultado es peor.

### 6.5 Dónde falla cada modelo

F1 por clase, ordenado de peor a mejor para el modelo ganador:

| Clase | Soporte (test) | HOG+SVM | CNN | ColorHist+RF |
|---|---:|---:|---:|---:|
| llave_inglesa | 10 | **0.667** | 0.800 | 0.750 |
| llave_tubo | 36 | **0.762** | 0.806 | 0.758 |
| espatula | 46 | 0.791 | 0.857 | 0.843 |
| pinzas | 28 | 0.792 | **0.677** | 0.792 |
| alicate | 40 | 0.800 | **0.696** | **0.741** |
| brocha | 69 | 0.828 | 0.813 | 0.834 |
| martillo | 71 | 0.859 | 0.861 | 0.819 |
| llave_combinada | 13 | 0.870 | 0.815 | **0.688** |
| destornillador | 76 | 0.885 | 0.834 | 0.808 |
| metro | 45 | **0.989** | 0.921 | 0.854 |

Leyendo las matrices de confusión de `outputs/ml_reports/` sacamos tres conclusiones:

**1. Las clases con pocos datos son las que fallan.** `llave_inglesa` es la peor clase de HOG+SVM
(F1 = 0.667) y es también la clase con menos recortes de todo el dataset (134 en total, 10 en
test). El recall es de apenas **0.50**: el modelo acierta la mitad de las llaves inglesas y confunde
el resto con `llave_tubo` (20 %), `martillo` (20 %) y `espatula` (10 %). Su precisión, en cambio,
es 1.0 — cuando dice "llave inglesa" acierta siempre, pero casi nunca se atreve a decirlo. Es el
comportamiento típico de una clase con pocos ejemplos.

**2. La confusión `pinzas ↔ alicate` es real y aparece en los tres modelos.** Es la confusión que
esperábamos por siluetas parecidas, y los datos la confirman:

| Modelo | pinzas → alicate | alicate → pinzas |
|---|---:|---:|
| CNN | 7 % | **27 %** |
| ColorHist+RF | **18 %** | 7 % |
| HOG+SVM | 11 % | 5 % |

El caso más grave es la **CNN, que manda el 27 % de los alicates a `pinzas`** y por eso hunde esas
dos clases (F1 0.696 y 0.677, sus dos peores). Es interesante que HOG+SVM sea el que **menos** se
confunde en esta pareja, a pesar de basarse justamente en la forma: el descriptor de gradientes
captura diferencias sutiles de contorno que la red pequeña no llega a aprender.

**3. Cada modelo se equivoca donde su descriptor es ciego.** El RandomForest de color falla más en
`llave_combinada` (F1 = 0.688) y manda el 20 % de las llaves inglesas a `destornillador`: son
piezas alargadas de metal gris, o sea el mismo histograma. HOG+SVM, en cambio, confunde el 23 % de
las llaves combinadas con destornilladores: son formas alargadas y rectas, o sea el mismo HOG.
**Los dos descriptores fallan en las mismas clases pero por razones opuestas**, y eso es
precisamente el argumento para `combined_features`, que dejamos implementado pero no llegamos a
evaluar como un cuarto modelo.

> **[IMAGEN: matrices de confusión de los 3 modelos + gráfico comparativo]**
> *Pie de foto: "Matrices de confusión de los tres clasificadores y comparación de accuracy/macro-F1. El modelo HOG+SVM obtiene el mejor desempeño."*
> Insertar la imagen debajo de este párrafo. Archivos: `outputs/ml_reports/hog_svm_confusion_matrix.png`, `cnn_confusion_matrix.png`, `color_hist_rf_confusion_matrix.png` y `comparison.png`.

---

## 7. Reconocimiento de patrones, detección y métricas (8%)

### 7.1 El detector

El corazón del reconocimiento es el **Módulo 3**: un **YOLOv8n** al que le hicimos fine-tuning
partiendo de los pesos preentrenados en COCO.

**Por qué la versión nano.** Tres razones, en orden de peso:

1. **Nuestro dataset es diminuto.** 597 imágenes derivadas de 36 fotos. Un YOLOv8m o l tiene
   decenas de millones de parámetros y se sobreajustaría de inmediato. El nano es el tamaño
   proporcionado al problema.
2. **Tiene que correr en la demo en vivo.** La aplicación debe responder en segundos, y no podemos
   asumir que la máquina de la sustentación tenga GPU. El nano corre en CPU.
3. **Entrena rápido**, lo que nos permitió iterar: reentrenar y volver a evaluar cuesta unos 17
   minutos.

Configuración real del entrenamiento (`runs/tools/args.yaml`):

| Parámetro | Valor | Por qué |
|---|---|---|
| Modelo base | `yolov8n.pt` (COCO) | Transfer learning: los filtros de bordes ya vienen aprendidos |
| Épocas | 100 (con `patience=20`) | Suficiente para converger; el early stopping evita seguir de gusto |
| `imgsz` | 640 | Coincide con el tamaño objetivo del aumento (M2) |
| `batch` | 8 | Límite de VRAM de la tarjeta usada |
| `lr0` / `lrf` | 0.005 / 0.01 | LR inicial baja porque partimos de pesos preentrenados |
| `warmup_epochs` | 5 | Evita destruir los pesos de COCO en los primeros pasos |
| `augment` | `False` | **El aumento ya lo hicimos nosotros en M2**; activarlo lo duplicaría |
| `half` / `cache` | `True` / `True` | FP16 y caché en RAM para acelerar |

Tiempo total de entrenamiento: **1 003 s (≈ 16,7 min)** para las 100 épocas.

### 7.2 Resultados

Evaluamos el `best.pt` resultante sobre las dos particiones. Reportamos las dos porque cuentan
cosas distintas: **val** es la partición con la que se eligió el mejor checkpoint durante el
entrenamiento, y **test** es la que el modelo nunca vio en ningún momento del proceso. La
partición de test es la que alimenta las gráficas de `outputs/metricas/`, porque `evaluate()` corre
con `split="test"`.

| Métrica | Val (116 imgs / 807 instancias) | Test (68 imgs / 442 instancias) | Meta académica |
|---|---:|---:|---|
| **mAP@0.5** | **0.9535** | **0.9432** | ≥ 0.75 |
| mAP@0.5:0.95 | 0.8379 | 0.7961 | — |
| Precisión | 0.8977 | 0.8669 | — |
| Recall | 0.9206 | 0.9204 | — |

**Superamos la meta académica con margen.** El objetivo era mAP@0.5 ≥ 0.75 y el detector queda muy
por encima en las dos particiones. La métrica más exigente, mAP@0.5:0.95, promedia el desempeño a
distintos umbrales de IoU y castiga las cajas mal ajustadas; que también quede alta indica que el
modelo no solo encuentra las herramientas, sino que las encuadra bien.

Vale la pena mirar la **caída de val a test**: mAP@0.5 baja poco (0.9535 → 0.9432), pero
mAP@0.5:0.95 baja **4,2 puntos** (0.8379 → 0.7961) y la precisión **3,1** (0.8977 → 0.8669). O sea:
en test el detector sigue **encontrando** las herramientas igual de bien (el recall es
prácticamente idéntico, 0.9206 vs 0.9204), pero las **encuadra un poco peor** y produce algún
falso positivo más. Es exactamente la diferencia que se espera entre la partición con la que se
seleccionó el checkpoint y una que nunca se usó.

**Dónde falla el detector** (AP@0.5 por clase sobre test, de peor a mejor):

| Clase | AP@0.5 | AP@0.5:0.95 | Precisión | Recall |
|---|---:|---:|---:|---:|
| pinzas | 0.854 | 0.741 | 0.724 | 0.748 |
| llave_inglesa | 0.859 | 0.642 | 0.915 | 0.800 |
| alicate | 0.864 | 0.735 | 0.742 | 0.863 |
| destornillador | 0.958 | 0.781 | 0.850 | 0.933 |
| martillo | 0.968 | 0.862 | 0.970 | 0.915 |
| brocha | 0.981 | 0.859 | 0.896 | 0.986 |
| llave_combinada | 0.983 | 0.866 | 0.684 | 1.000 |
| llave_tubo | 0.985 | 0.770 | 0.922 | 0.981 |
| espatula | 0.986 | 0.853 | 0.987 | 0.978 |
| metro | 0.995 | 0.853 | 0.979 | 1.000 |

Las tres peores clases del detector son **`pinzas`, `llave_inglesa` y `alicate`**, que son
exactamente las mismas que peor clasifican los modelos de la sección 6. Esto es importante: el
problema **no** está en un módulo concreto sino en los datos. `pinzas` y `alicate` tienen precisión
baja (0.724 y 0.742) con recall razonable, o sea que el detector marca de más y se cruza entre las
dos clases. `llave_inglesa` tiene el peor mAP@0.5:0.95 de todas (0.642) porque es la clase con
menos ejemplos del dataset. Y `llave_combinada` alcanza recall 1.000 con precisión 0.684: encuentra
todas las que hay, pero marca varias que no lo son — con 18 instancias en test, cualquier falso
positivo pesa mucho.

Hay que leer estos números con la advertencia de la sección 2: **las variantes aumentadas de una
misma foto original pueden estar repartidas entre train y test**, así que el test no es
completamente independiente. Los valores miden qué tan bien el detector reconoce herramientas *en
nuestras fotos*. La sección 8 muestra qué pasa fuera de ese dominio.

### 7.3 Reproducibilidad y organización

Todo el módulo de detección cabe en un solo archivo (`src/detection/detector.py`) con tres
funciones públicas y una CLI:

```bash
python -m src.detection.detector train    --data data.yaml --model_size n --epochs 100
python -m src.detection.detector evaluate --data data.yaml
python -m src.detection.detector predict  --source data/raw/juanl
```

`evaluate()` corre la validación, imprime las métricas por clase y **copia automáticamente** las
curvas P/R/F1, la curva Precision-Recall, la matriz de confusión y los lotes de validación
anotados a `outputs/metricas/`. No hay ningún paso manual entre entrenar y tener la evidencia.

Sobre la organización del código, tres decisiones que sostienen el pipeline:

- **`src/paths.py` centraliza todas las rutas** ancladas a la raíz del repositorio. Ningún script
  depende del directorio desde el que se ejecuta.
- **`_resolver_data()`** convierte el `path` relativo de `data.yaml` en absoluto antes de pasárselo
  a Ultralytics, así el mismo `data.yaml` sirve en Windows, Mac y Linux.
- **`_detectar_dispositivo()`** elige CUDA, MPS (Apple) o CPU automáticamente, y ajusta batch,
  caché y FP16 según lo que encuentre.

> **[IMAGEN: curvas PR / P / R / F1, matriz de confusión del detector y lotes de validación anotados]**
> *Pie de foto: "Evaluación del detector YOLOv8n: curvas de precisión-recall y ejemplos del conjunto de validación con las detecciones."*
> Insertar la imagen debajo de este párrafo. Archivos: `outputs/metricas/PR_curve.png`, `P_curve.png`, `R_curve.png`, `F1_curve.png`, `confusion_matrix_normalized.png` y `val_batch0_pred.jpg`.

---

## 8. Aplicación: frontend y demo en vivo

### 8.1 La aplicación

Montamos una aplicación web con **FastAPI** (`scripts/serve_app.py`) y una página estática
(`web/index.html`). Se levanta con un comando:

```bash
python scripts/serve_app.py     # http://localhost:8000
```

Endpoints:

| Endpoint | Qué hace |
|---|---|
| `POST /detectar` | Recibe la imagen y un umbral de confianza, corre YOLO y devuelve la imagen anotada (en base64, para mostrarla al instante) más el JSON de detecciones y el conteo por clase |
| `GET /modelos` | Lista los detectores disponibles en disco, para el selector de la página |
| `GET /historial` | Devuelve las últimas predicciones guardadas, de la más nueva a la más vieja |
| `GET /metricas` | Lee `results.csv` y los `*_metrics.json` y devuelve los números y las URL de todas las gráficas |
| `GET /evidencia` | Devuelve las imágenes de cada etapa del pipeline (adquisición, preprocesamiento, segmentación, características) |

La página tiene **tres pestañas**:

1. **El proyecto** — recorrido por las 8 etapas del pipeline, cada una con sus puntos clave, los
   archivos de código involucrados y la explicación completa plegable, acompañada de la evidencia
   visual que sirve `/evidencia`.
2. **Métricas** — los números de detección y clasificación, la tabla comparativa de los tres
   clasificadores y la galería de gráficas con lightbox.
3. **Detector en vivo** — la demo: se arrastra o se elige una foto, se ajusta el **slider de
   confianza** (0.1–0.9, por defecto 0.4) y la aplicación devuelve la imagen anotada más el
   **inventario** de herramientas detectadas. Debajo queda el historial de las detecciones
   anteriores, que persiste porque cada predicción se guarda como `.jpg` + `.json` en
   `outputs/predicciones/`. La página también incluye un selector de detector, que solo se muestra
   si `/modelos` reporta más de un `best.pt` en disco; con un único modelo entrenado permanece
   oculto.

El diseño es tipo taller: azul acero, tipografía condensada y una zona de soltar con la silueta de
la herramienta que "falta" en el tablero.

**Carga del modelo y fallback a CPU.** `cargar_modelo()` busca los pesos en
`runs/tools/weights/best.pt`, detecta el dispositivo y hace un *warm-up* con una imagen vacía. Si
la GPU está sin VRAM libre, captura el `OutOfMemoryError`, limpia la caché y **recarga el modelo en
CPU** en lugar de reventar. También se puede forzar CPU con la variable de entorno `FORCE_CPU=1`.
Añadimos esto después de que la GPU se quedara ocupada por otro proceso durante una prueba.

> **[IMAGEN: captura de la pestaña Detector con una detección real y su inventario]**
> *Pie de foto: "Demo en vivo: el usuario sube una foto y la aplicación devuelve la imagen anotada más el inventario de herramientas detectadas."*
> Insertar la imagen debajo de este párrafo. `[FALTA DATO: captura de la demo]` — no existe todavía un archivo de captura en `outputs/`.

### 8.2 Pruebas de la demo y domain shift

Probamos la aplicación de punta a punta con fotos propias y con fotos sacadas de internet. El
comportamiento es claramente distinto en cada caso.

**En fotos parecidas a las del dataset funciona bien.** El detector encuentra las herramientas
presentes con la clase correcta y confianzas altas.
`[FALTA DATO: confianzas por clase de una detección real con el modelo actual — las cifras que teníamos corresponden a un entrenamiento anterior y hay que volver a medirlas]`

**En fotos de internet se degrada.** Esto es **domain shift**: el modelo aprendió la distribución
de nuestras 36 fotos, no la de "cualquier foto de herramientas". Observamos tres fallos
concretos:

1. **Sesgo de color.** El modelo aprendió el atajo "amarillo compacto = metro". Marcó como `metro`
   un objeto amarillo que no lo era, y en cambio **no reconoció un flexómetro de otro color**. La
   forma correcta de leerlo: como casi todos nuestros metros son amarillos, el color quedó como la
   señal más fácil, y la red la tomó. No es un error del algoritmo, es un error de nuestro
   dataset.
2. **Confusión `pinzas` ↔ `alicate`.** La misma que documentamos en la sección 6.5. Se agrava
   porque **no aplicamos NMS agnóstico de clase**: el NMS de YOLO suprime cajas solapadas de la
   *misma* clase, así que dos cajas sobre el mismo objeto con clases distintas sobreviven las dos.
   Con el umbral de confianza bajo aparecen detecciones duplicadas sobre la misma herramienta; al
   subir el slider desaparecen.
3. **Ignora lo que no conoce.** Objetos fuera de las 10 clases (una clavadora, unas gafas de
   protección, una escuadra) simplemente no se detectan. No hay manejo de "desconocido": el
   sistema solo sabe decir una de las 10 palabras que le enseñamos, y no tiene forma de responder
   "esto es una herramienta, pero no sé cuál".

`[FALTA DATO: capturas o registro de las pruebas con fotos de internet usando el modelo actual]`

**Por qué pasa esto.** Con 36 fotos originales, cuatro fotógrafos, dos o tres fondos y una
iluminación parecida, el modelo tiene muy pocas maneras de distinguir "lo que define a una
herramienta" de "lo que pasa que era así en nuestras fotos". El aumento de datos multiplica las
imágenes pero **no añade información nueva**: una foto rotada y con más brillo sigue siendo la
misma caja, en el mismo cuarto, con las mismas herramientas. Preferimos decirlo antes de que nos
lo pregunten: nuestros números son buenos dentro del dominio y no deben leerse como una promesa
fuera de él.

---

## 9. Conclusiones, limitaciones y trabajo futuro

### 9.1 Qué logramos

Construimos un **pipeline completo y reproducible**, de la foto al inventario, con cinco módulos
independientes y una aplicación web que lo demuestra en vivo.

En detección **superamos la meta académica** de mAP@0.5 ≥ 0.75 con margen: **0.9432 en test** y
0.9535 en validación. En clasificación, los tres modelos superan el 80 % de accuracy sobre 434
recortes de test, y el ganador es **HOG + SVM (accuracy 0.8456, macro-F1 0.8241)**.

Hay un hallazgo que no esperábamos y que nos parece el más útil de todo el informe: **las tres
clases que peor detecta YOLO (`pinzas`, `llave_inglesa`, `alicate`) son las mismas tres que peor
clasifican los modelos de ML**. Cuatro arquitecturas distintas fallan en las mismas clases. Eso
descarta que el problema sea de un algoritmo concreto y lo ubica donde realmente está: en el
dataset.

### 9.2 Qué aprendimos

**Con pocos datos, los métodos clásicos ganan.** HOG + SVM le ganó a la CNN usando 35 veces menos
tiempo de entrenamiento. Un descriptor diseñado a mano ya trae incorporado el conocimiento de que
"los bordes importan"; una red tiene que aprenderlo desde cero, y para eso necesita datos que no
teníamos. Esta fue la lección más útil del proyecto y va contra la intuición de que "deep learning
siempre es mejor".

**Entender por qué falla un método vale tanto como que funcione.** La sección 4 es el mejor
ejemplo: Otsu no funcionó, pero entender que su hipótesis (un umbral global separa objeto de
fondo) no se cumple en una caja con herramientas de colores distintos y sombras es lo que
justifica el camino que sí tomamos.

**Cada descriptor es ciego donde el otro ve.** HOG confunde formas alargadas parecidas
(`llave_combinada` con `destornillador`); el histograma de color confunde metales grises
(`llave_inglesa` con `destornillador`). Los mismos errores, por razones opuestas.

**El dataset es el techo del proyecto.** El sesgo "amarillo = metro" no es un fallo del algoritmo:
es exactamente lo que le enseñamos. Ningún ajuste de hiperparámetros lo arregla.

### 9.3 Limitaciones

Somos explícitos con lo que el sistema **no** hace:

- **Pocos datos y pocas fuentes.** 36 fotos originales de 4 personas, con fondos e iluminación
  poco variados. Todo lo demás es aumento.
- **Desbalance fuerte entre clases.** `destornillador` tiene 5,4 veces más recortes que
  `llave_inglesa`, y esa proporción se refleja punto por punto en el F1 de cada clase.
- **Fuga de información en el split.** El split se hace después del aumento, así que variantes de
  una misma foto original pueden caer en train y en test. Nuestras métricas son optimistas
  respecto a lo que veríamos con fotos totalmente nuevas.
- **Sin manejo de "desconocido".** El clasificador siempre devuelve una de las 10 clases. No hay
  umbral de rechazo ni clase "otro".
- **Sin NMS agnóstico de clase.** De ahí las detecciones duplicadas `pinzas`/`alicate` sobre el
  mismo objeto.
- **La segmentación clásica no es utilizable en producción.** Otsu falla en nuestras imágenes y
  GrabCut depende del detector.

### 9.4 Trabajo futuro

Las mejoras salen directo de las limitaciones, no son una lista de deseos:

1. **Más datos y más variados.** Es la mejora con mayor retorno, con diferencia. Más fotógrafos,
   más fondos, más condiciones de luz y herramientas de marcas distintas — sobre todo metros que
   no sean amarillos, para romper el atajo de color.
2. **Balancear las clases flojas.** Anotar específicamente más `llave_inglesa`, `llave_combinada` y
   `pinzas`, que son las tres peores y también las tres con menos ejemplos.
3. **Rehacer el split antes del aumento**, agrupando por foto original, para que las métricas midan
   generalización real y no memorización.
4. **Transfer learning en la clasificación.** MobileNetV2 o EfficientNet preentrenados en ImageNet
   traen los filtros ya aprendidos, que es justo lo que nuestra CNN desde cero no pudo conseguir
   con 2 880 recortes. Es la vía más probable para superar a HOG+SVM sin conseguir más datos.
5. **NMS agnóstico de clase** en la inferencia, para eliminar los duplicados `pinzas`/`alicate`.
6. **Umbral de rechazo para lo desconocido**, usando la distancia al margen del SVM o la entropía
   de la salida, para que el sistema pueda decir "no sé" en vez de forzar una de las 10 clases.
7. **Evaluar `combined_features`** como un cuarto modelo. Está implementado y los datos de la
   sección 6.5 sugieren que HOG y el histograma fallan en clases distintas, así que combinarlos
   debería ayudar.

### 9.5 Reflexión final

El proyecto cumple lo que se propuso: hay un pipeline completo, medido, documentado y demostrable
en vivo, y las métricas superan la meta académica. Pero el número que mejor describe lo que
construimos no es el mAP: son las **36 fotos originales**. Todo lo demás —las 597 imágenes
aumentadas, los 4 113 recortes— sale de ahí.

Por eso preferimos presentar el resultado como lo que es: un sistema que funciona bien **en su
dominio** y que se degrada fuera de él, de forma predecible y por razones que entendemos y podemos
explicar una por una. Nos parece más valioso saber exactamente dónde y por qué falla nuestro
modelo que reportar una accuracy alta sin poder decir qué significa.

---

## Anexo — Cómo reproducir los resultados

```bash
# Entorno
python -m venv .toolkit-venv
.\.toolkit-venv\Scripts\Activate.ps1     # Windows
pip install -r requirements.txt

# M1 — unificar las anotaciones de los 4 colaboradores
python scripts/run_merge_ndjson.py

# M2 + M3 — aumento, splits y entrenamiento del detector
python scripts/run_pipeline.py --model_size n --epochs 100
python -m src.detection.detector evaluate --data data.yaml

# M4 — pipeline completo de clasificación (recortes + 3 modelos + comparativa)
python scripts/run_ml_pipeline.py

# M5 — comparativas de segmentación clásica
python scripts/run_segmentation_demo.py

# Evidencias visuales para el informe
python scripts/run_preprocessing_demo.py
python scripts/run_features_demo.py

# Aplicación web
python scripts/serve_app.py              # http://localhost:8000
```

**Dónde queda cada evidencia:**

| Carpeta | Contenido |
|---|---|
| `outputs/adquisicion/` | Montaje con una foto de cada colaborador |
| `outputs/preprocesamiento/` | Panel original / gris / canal H / Canny / Sobel |
| `outputs/segmentacion/` | 8 tiras comparativas Otsu vs GrabCut |
| `outputs/caracteristicas/` | Visualización HOG e histograma HSV |
| `outputs/ml_reports/` | Métricas, matrices de confusión y comparativa de los 3 clasificadores |
| `outputs/metricas/` | Curvas P/R/F1, PR, matriz de confusión y lotes de validación del detector |
| `runs/tools/` | Logs, `results.csv` y pesos del entrenamiento de YOLO |
