# Project Walkthrough: Clasificador de imágenes de perros y gatos

## 1. Introducción al proyecto

### Objetivo

El objetivo de este proyecto es desarrollar un modelo de clasificación de imágenes capaz de identificar automáticamente si una imagen contiene un gato o un perro.

Para ello, se utiliza una red neuronal convolucional basada en **EfficientNetB0**, una arquitectura de aprendizaje profundo diseñada para el reconocimiento de imágenes.

El modelo se entrena con un conjunto de imágenes etiquetadas y posteriormente se evalúa para medir su capacidad de clasificación. También se implementa un script que permite introducir una imagen individual y obtener una predicción.

### Tecnologías utilizadas

- Python 3.13
- TensorFlow 2.21
- Keras 3.15
- EfficientNetB0
- NumPy
- Matplotlib
- Scikit-learn
- Git y GitHub
- GitHub Codespaces

### Estructura general del proyecto

El proyecto se organiza en los siguientes directorios:

- `data/`: contiene los datos utilizados para entrenar y evaluar el modelo.
- `docs/`: contiene la documentación del proyecto.
- `models/`: almacena el modelo entrenado.
- `src/`: contiene los scripts de preparación de datos, exploración, entrenamiento, evaluación y predicción.

## 2. Configuración del entorno de trabajo

### Entorno de desarrollo

El proyecto se desarrolló utilizando **GitHub Codespaces**, un entorno de desarrollo en la nube que permite trabajar directamente sobre un repositorio de GitHub.

Se utilizó un entorno virtual de Python (`.venv`) para mantener las dependencias del proyecto aisladas del resto del sistema.

### Versiones principales

Durante el desarrollo se utilizaron las siguientes versiones:

| Herramienta | Versión |
|---|---|
| Python | 3.13.18 |
| TensorFlow | 2.21.0 |
| Keras | 3.15.1 |

### Dependencias

Las dependencias del proyecto se registran en el archivo `requirements.txt`.

Para instalar los paquetes, se utiliza:

```bash
pip install -r requirements.txt
```

### Consideraciones del entorno

El entrenamiento se realizó utilizando CPU, ya que el entorno de Codespaces no disponía de una GPU CUDA.

Además, debido a las limitaciones de memoria RAM del entorno, fue necesario ajustar el tamaño de los lotes (*batch size*) durante el entrenamiento. Esta decisión se explica con más detalle en la sección correspondiente al entrenamiento del modelo.

## 3. Descarga y preparación del dataset

### 3.1. Descarga de los datos

Para desarrollar el clasificador se utilizó el dataset **Dogs vs. Cats**, proporcionado por Google Cloud Storage.

Fuente: [Dogs vs. Cats Dataset](https://storage.googleapis.com/datascience-materials/dogs-vs-cats.zip)

El archivo descargado fue:

```text
data/dogs-vs-cats.zip
```

Después de extraerlo, se obtuvo un directorio con 25.000 imágenes JPG:

```text
data/
└── raw/
    └── dogs-vs-cats/
        └── train/
            ├── cat.0.jpg
            ├── cat.1.jpg
            ├── ...
            ├── dog.0.jpg
            ├── dog.1.jpg
            └── ...
```

El conjunto original contiene:

- 12.500 imágenes de gatos.
- 12.500 imágenes de perros.

### 3.2. Organización de los datos

Para preparar los datos, se creó el script:

```text
src/prepare_dataset.py
```

El objetivo es dividir las imágenes en tres conjuntos:

- **Training set:** 80 % de las imágenes.
- **Validation set:** 10 % de las imágenes.
- **Test set:** 10 % de las imágenes.

La división se realizó por separado para cada categoría, manteniendo la misma proporción de imágenes de gatos y perros.

Se utilizó una semilla aleatoria (`random.seed(42)`) para que la selección de imágenes fuera reproducible.

### 3.3. Distribución final

El resultado de la preparación fue el siguiente:

| Categoría | Training | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Cats | 10.000 | 1.250 | 1.250 | 12.500 |
| Dogs | 10.000 | 1.250 | 1.250 | 12.500 |
| **Total** | **20.000** | **2.500** | **2.500** | **25.000** |

La estructura final quedó organizada de esta manera:

```text
data/
└── processed_split/
    ├── train/
    │   ├── cats/
    │   └── dogs/
    ├── validation/
    │   ├── cats/
    │   └── dogs/
    └── test/
        ├── cats/
        └── dogs/
```

### 3.4. Optimización del almacenamiento

Para organizar las imágenes se utilizó `os.link()`, que crea enlaces físicos (*hard links*) a los archivos originales.

Esta solución permitió separar las imágenes en los directorios de entrenamiento y prueba sin duplicar físicamente los archivos, reduciendo así el consumo de almacenamiento.

### 3.5. Resultado

El script de preparación crea los tres conjuntos en `data/processed_split/`. Los archivos existentes en la ruta anterior `data/processed/` se conservan sin cambios.

## 4. Exploración y visualización de imágenes

### 4.1. Objetivo

Antes de entrenar el modelo, se realizó una exploración visual del dataset para comprobar el contenido de las imágenes y familiarizarse con los datos que utilizaría el clasificador.

### 4.2. Herramientas utilizadas

Para esta etapa se creó el notebook:

```text
src/explore.ipynb
```

Se utilizaron las siguientes librerías:

- `os`: para acceder a los archivos y directorios.
- `Matplotlib`: para visualizar las imágenes.
- `TensorFlow Keras`: para cargar las imágenes.

Importaciones utilizadas:

```python
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
```

### 4.3. Visualización de las imágenes

Se visualizaron las primeras nueve imágenes de cada categoría por separado:

- Nueve imágenes de gatos.
- Nueve imágenes de perros.

Esto permitió comprobar visualmente las características de las imágenes y verificar que los archivos estaban correctamente organizados por categoría.

### 4.4. Resultado

La exploración visual se completó correctamente. Se pudieron visualizar ambas categorías y comprobar que el dataset contenía imágenes de gatos y perros adecuadas para continuar con el proceso de clasificación.

## 5. Carga de datos con ImageDataGenerator

### 5.1. Objetivo

Una vez preparado el dataset, fue necesario establecer un mecanismo para cargar las imágenes y suministrarlas al modelo durante el entrenamiento y la evaluación.

Para ello, se utilizó `ImageDataGenerator` de TensorFlow Keras, que permite cargar las imágenes progresivamente mediante lotes (*batches*).

### 5.2. Configuración

Se creó el archivo:

```text
src/data_loader.py
```

Se utilizó la siguiente importación:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

Se definieron los siguientes parámetros:

```python
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8

dataset_dir = "data/processed_split"
train_dir = f"{dataset_dir}/train"
validation_dir = f"{dataset_dir}/validation"
test_dir = f"{dataset_dir}/test"
```

Las imágenes se redimensionaron a 224 × 224 píxeles para que coincidieran con las dimensiones de entrada establecidas posteriormente en la arquitectura EfficientNetB0.

El tamaño del lote se estableció inicialmente en 8 imágenes para reducir el consumo de memoria durante el procesamiento.

### 5.3. Creación de los generadores

Se crearon dos generadores independientes:

- `trdata`: encargado de cargar las imágenes de entrenamiento.
- `valdata`: encargado de cargar las imágenes de validación.
- `tsdata`: encargado de cargar las imágenes de prueba, reservadas para la evaluación final.

Ambos utilizan `ImageDataGenerator()` sin transformaciones adicionales.

Se configuró `class_mode="categorical"` porque el modelo realiza una clasificación de dos categorías mediante una capa de salida Softmax.

También se estableció:

- `shuffle=True` para mezclar las imágenes durante el entrenamiento.
- `shuffle=False` para mantener el orden de las imágenes durante la evaluación y poder comparar correctamente las predicciones con las etiquetas reales.

### 5.4. Carga de las imágenes

Los generadores utilizan `flow_from_directory()` para localizar automáticamente las imágenes y asignarles sus respectivas etiquetas a partir de la estructura de carpetas.

El resultado de la carga confirma los tamaños esperados:

```text
Found 20000 images belonging to 2 classes.
Found 2500 images belonging to 2 classes.
Found 2500 images belonging to 2 classes.
```

Por tanto, se confirmó que los generadores reconocían correctamente las dos categorías y sus respectivas imágenes.

### 5.5. Consideraciones sobre el preprocesamiento

No se añadió una transformación `rescale=1./255` al generador, ya que EfficientNetB0 incorpora su propio preprocesamiento de entrada.

Se mantuvo así la configuración compatible con la arquitectura seleccionada.

### 5.6. Resultado

La carga de datos se completó correctamente:

- 20.000 imágenes disponibles para entrenamiento.
- 2.500 imágenes disponibles para validación.
- 2.500 imágenes reservadas para la evaluación final.
- Imágenes redimensionadas a 224 × 224 píxeles.
- Procesamiento por lotes de 8 imágenes.

Los generadores `trdata`, `valdata` y `tsdata` quedaron preparados para entrenamiento, selección del modelo y evaluación final, respectivamente.

## 6. Construcción de la arquitectura EfficientNetB0

### 6.1. Selección de la arquitectura

Para desarrollar el clasificador se seleccionó **EfficientNetB0**, una arquitectura de red neuronal convolucional diseñada para tareas de reconocimiento y clasificación de imágenes.

Se utilizó la implementación disponible en Keras mediante `keras.applications`.

La arquitectura se configuró con `include_top=False`, lo que permite utilizar la base convolucional de EfficientNetB0 sin su clasificador original.

Además, se estableció `weights=None`, por lo que el modelo comienza el entrenamiento desde cero, sin utilizar pesos preentrenados.

### 6.2. Implementación del modelo

Se creó el archivo:

```text
src/model.py
```

El modelo se construyó utilizando una arquitectura secuencial (`Sequential`) con las siguientes capas:

1. **EfficientNetB0:** extrae características relevantes de las imágenes.
2. **GlobalAveragePooling2D:** reduce las dimensiones espaciales de las características extraídas.
3. **Dense (128 neuronas, ReLU):** procesa las características para realizar la clasificación.
4. **Dense (2 neuronas, Softmax):** produce las probabilidades correspondientes a las dos categorías.

### 6.3. Configuración de entrada

Se estableció una dimensión de entrada de:

```python
input_shape=(224, 224, 3)
```

Esto corresponde a imágenes de 224 × 224 píxeles con tres canales de color (RGB).

### 6.4. Compilación del modelo

Para configurar el proceso de aprendizaje se utilizaron los siguientes parámetros:

| Parámetro | Configuración |
|---|---|
| Optimizador | Adam |
| Función de pérdida | Categorical Crossentropy |
| Métrica | Accuracy |
| Función de activación de la capa oculta | ReLU |
| Función de activación de salida | Softmax |

Se seleccionó `categorical_crossentropy` porque las etiquetas se proporcionan en formato categórico mediante los generadores de datos.

### 6.5. Resumen de la arquitectura

Al ejecutar `model.summary()`, se obtuvo la siguiente estructura:

| Capa | Salida | Parámetros |
|---|---|---:|
| EfficientNetB0 | (None, 7, 7, 1280) | 4.049.571 |
| GlobalAveragePooling2D | (None, 1280) | 0 |
| Dense | (None, 128) | 163.968 |
| Dense | (None, 2) | 258 |

**Parámetros totales:** 4.213.797.

De ellos, 4.171.774 son entrenables y 42.023 no entrenables.

### 6.6. Resultado

La arquitectura se construyó y compiló correctamente. El modelo quedó preparado para aprender a distinguir entre imágenes de gatos y perros durante la siguiente etapa de entrenamiento.

## 7. Entrenamiento del modelo

### 7.1. Preparación del entrenamiento

Para entrenar la red neuronal se creó el archivo:

```text
src/train.py
```

Se importaron los generadores de datos (`trdata` y `valdata`), el modelo definido anteriormente y dos callbacks de Keras:

- `ModelCheckpoint`
- `EarlyStopping`

### 7.2. Configuración de ModelCheckpoint

Se utilizó `ModelCheckpoint` para guardar automáticamente el mejor modelo durante el entrenamiento.

La configuración fue:

| Parámetro | Valor |
|---|---|
| Archivo | `models/best_model.keras` |
| Monitorización | `val_accuracy` |
| `save_best_only` | `True` |
| Modo | `max` |

De esta manera, el archivo conserva los pesos correspondientes a la época con mayor precisión de validación.

### 7.3. Configuración de EarlyStopping

También se implementó `EarlyStopping` para controlar el proceso de entrenamiento y evitar continuar innecesariamente cuando la pérdida de validación deja de mejorar.

Su configuración fue:

| Parámetro | Valor |
|---|---|
| Monitorización | `val_loss` |
| Patience | 3 |
| `restore_best_weights` | `True` |

Al finalizar, el callback restauró los pesos correspondientes a la época con menor pérdida de validación.

### 7.4. Parámetros de entrenamiento

Se utilizó `model.fit()` para entrenar el modelo con los siguientes parámetros:

- Generador de entrenamiento: `trdata`.
- Datos de validación: `valdata`.
- Número máximo de épocas: 10.
- Batch size: 8 imágenes.
- Callbacks: `ModelCheckpoint` y `EarlyStopping`.

### 7.5. Problemas de memoria y ajustes realizados

Durante las primeras pruebas se utilizó un batch size de 32 imágenes. Sin embargo, el proceso de entrenamiento fue terminado por el entorno, probablemente debido a las limitaciones de memoria RAM disponibles en Codespaces.

Para solucionar este problema, se redujo el batch size a 8 imágenes.

Con este ajuste, el entrenamiento pudo completarse correctamente.

Además, el entorno no disponía de GPU CUDA, por lo que el entrenamiento se realizó utilizando CPU. Cada época tardó aproximadamente entre 44 y 49 minutos.

### 7.6. Resultados del entrenamiento

La tabla siguiente se conserva del entrenamiento anterior, que utilizó incorrectamente el antiguo conjunto de prueba como validación. Son cifras históricas y no describen el nuevo split:

| Época | Accuracy | Loss | Val. Accuracy | Val. Loss |
|---:|---:|---:|---:|---:|
| 1 | 59,15 % | 0,6762 | 58,26 % | 0,6691 |
| 2 | 68,83 % | 0,5900 | 72,60 % | 0,5391 |
| 3 | 77,57 % | 0,4809 | 82,28 % | 0,3982 |
| 4 | 82,82 % | 0,3889 | 84,68 % | 0,3535 |
| 5 | 86,25 % | 0,3210 | 88,12 % | 0,2820 |
| 6 | 89,47 % | 0,2581 | 88,32 % | 0,2881 |
| 7 | 91,17 % | 0,2183 | 90,34 % | 0,2397 |
| 8 | 92,56 % | 0,1858 | 91,10 % | 0,2306 |
| 9 | 93,67 % | 0,1603 | 91,32 % | 0,2307 |
| 10 | 94,48 % | 0,1395 | 91,12 % | 0,2366 |

### 7.7. Selección del mejor modelo

En el entrenamiento anterior, `ModelCheckpoint` guardó la época 9 con una accuracy de validación del 91,32 %. Esos datos de validación también eran el antiguo conjunto de prueba, así que esta cifra no es una métrica de prueba independiente ni corresponde a una nueva ejecución con el split corregido.

En aquella ejecución, `EarlyStopping` monitorizaba `val_loss` y restauró los pesos de la época 8, que obtuvo la menor pérdida de validación: 0,2306. El archivo `models/best_model.keras` existente también procede de la ejecución anterior; vuelve a entrenar con el split corregido antes de evaluarlo.

### 7.8. Consideración sobre los datos de validación

El flujo corregido utiliza `valdata` mediante el parámetro `validation_data`. `tsdata` no se emplea durante el entrenamiento ni para seleccionar el checkpoint, y queda reservado para la evaluación final. El 91,32 % reportado anteriormente procede del antiguo conjunto de prueba reutilizado y no debe interpretarse como una métrica independiente. Para obtener nuevas métricas de test es necesario volver a entrenar con los tres conjuntos.

### 7.9. Resultado

El entrenamiento finalizó correctamente y se generó el archivo:

```text
models/best_model.keras
```

La nueva ejecución de entrenamiento guardará en esta ruta el modelo seleccionado por `ModelCheckpoint`. El archivo que ya existe en el workspace corresponde a la ejecución anterior y debe reemplazarse antes de la evaluación final.

## 8. Evaluación del modelo

### 8.1. Objetivo

Una vez finalizado el entrenamiento, se evaluó el modelo para medir su capacidad de clasificación y comprobar qué tan bien identificaba las imágenes de gatos y perros.

Para ello, se utilizan el modelo guardado mediante `ModelCheckpoint` y el generador `tsdata`.

### 8.2. Implementación

Se creó el archivo:

```text
src/evaluate.py
```

En este script se utilizaron las siguientes herramientas:

- `load_model`: para cargar el modelo entrenado.
- `model.evaluate()`: para calcular la pérdida y la precisión.
- `model.predict()`: para generar predicciones sobre las imágenes.
- `numpy.argmax()`: para convertir las probabilidades en etiquetas de clase.
- `accuracy_score`: para calcular la precisión.
- `confusion_matrix`: para generar la matriz de confusión.
- `classification_report`: para obtener las métricas de clasificación.

### 8.3. Métricas generales

La ejecución anterior reportó una accuracy del 91,32 %, pero evaluó las mismas imágenes usadas para validación y selección del modelo. Esa cifra no es una medición de prueba independiente y se retiró de los resultados actuales. Ejecuta `src/train.py` y después `src/evaluate.py` para generar métricas sobre el conjunto de prueba reservado de 2.500 imágenes.

### 8.4. Resultado

El script de evaluación calcula loss, accuracy, matriz de confusión e informe de clasificación únicamente para `tsdata`. No hay un resultado independiente disponible hasta volver a entrenar con el split corregido.

## 9. Predicciones individuales

### 9.1. Objetivo

Además de evaluar el modelo sobre el conjunto de prueba, se implementó una funcionalidad que permite introducir una imagen individual y obtener una predicción.

Esto permite comprobar el funcionamiento del clasificador con imágenes seleccionadas manualmente.

### 9.2. Implementación

Se creó el archivo:

```text
src/predict.py
```

El script realiza las siguientes operaciones:

1. Carga el modelo entrenado desde `models/best_model.keras`.
2. Solicita al usuario la ruta de una imagen.
3. Carga la imagen utilizando `load_img()`.
4. Redimensiona la imagen a 224 × 224 píxeles.
5. Convierte la imagen en un array numérico mediante `img_to_array()`.
6. Añade una dimensión adicional para crear un lote de una imagen utilizando `numpy.expand_dims()`.
7. Ejecuta `model.predict()` para obtener las probabilidades de clasificación.
8. Utiliza `numpy.argmax()` para seleccionar la categoría con mayor probabilidad.
9. Muestra la categoría predicha y el porcentaje de confianza.

### 9.3. Ejecución

Desde la raíz del proyecto se puede ejecutar:

```bash
python src/predict.py
```

El programa solicita la ruta de la imagen:

```text
Enter the image path:
```

El usuario introduce la ruta correspondiente y el modelo devuelve el resultado.

### 9.4. Pruebas realizadas

Se realizaron pruebas con dos imágenes del dataset original:

| Imagen | Categoría predicha | Confianza |
|---|---|---:|
| `cat.10.jpg` | Cats | 98,88 % |
| `dog.10.jpg` | Dogs | 99,26 % |

Ambas imágenes fueron clasificadas correctamente.

### 9.5. Resultado

La funcionalidad de predicción individual se implementó correctamente y permite utilizar el modelo entrenado para clasificar imágenes de gatos y perros de forma independiente.

El porcentaje mostrado representa la probabilidad asignada por el modelo a la categoría predicha, no una garantía de que la clasificación sea correcta.

## 10. Problemas encontrados y soluciones

Durante el desarrollo del proyecto surgieron diferentes dificultades relacionadas con el entorno de ejecución, las dependencias y los recursos disponibles.

### 10.1. Limitaciones de memoria RAM

**Problema:**

Durante las primeras pruebas de entrenamiento, se utilizó un batch size de 32 imágenes. El proceso fue terminado por el entorno de Codespaces, probablemente debido al consumo de memoria.

**Solución:**

Se redujo el batch size a 8 imágenes para disminuir el consumo de memoria y permitir que el entrenamiento pudiera completarse correctamente.

### 10.2. Compatibilidad de ImageDataGenerator con Keras

**Problema:**

La importación de `ImageDataGenerator` desde `keras.preprocessing.image` no estaba disponible en la versión de Keras instalada.

**Solución:**

Se utilizó la importación desde TensorFlow Keras:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

Esto permitió crear los generadores de entrenamiento y evaluación sin problemas.

### 10.3. Uso de fit_generator()

**Problema:**

Durante el desarrollo se identificó que `fit_generator()` ya no era necesario en la versión de Keras utilizada.

**Solución:**

Se utilizó `model.fit()` directamente con los generadores de imágenes, ya que permite entrenar el modelo utilizando este tipo de entrada.

### 10.4. Entrenamiento sin GPU

**Problema:**

El entorno de GitHub Codespaces utilizado no disponía de una GPU CUDA compatible.

**Solución:**

El entrenamiento se realizó utilizando CPU. Aunque esto incrementó el tiempo de procesamiento, permitió completar las 10 épocas sin necesidad de cambiar de infraestructura.

### 10.5. Optimización del almacenamiento del dataset

**Problema:**

La organización del dataset requería separar las imágenes originales en directorios de entrenamiento y prueba, lo que podía generar duplicación de archivos y un consumo innecesario de almacenamiento.

**Solución:**

Se utilizó `os.link()` para crear enlaces físicos (*hard links*) a las imágenes originales, evitando duplicar físicamente los archivos.

### 10.6. Gestión de archivos y dependencias

Se utilizó un entorno virtual de Python para aislar las dependencias y un archivo `requirements.txt` para registrar los paquetes necesarios.

También se configuró `.gitignore` para excluir los datasets, archivos ZIP y modelos entrenados del control de versiones.

Esto permitió mantener el repositorio más ligero y evitar subir archivos de gran tamaño a GitHub.

## 11. Resultados finales y conclusiones

### 11.1. Objetivos alcanzados

Se completó el desarrollo de un clasificador binario de imágenes capaz de distinguir entre gatos y perros mediante EfficientNetB0.

Durante el proyecto se realizaron las siguientes tareas:

- Descarga y organización de un dataset de 25.000 imágenes.
- División de los datos en conjuntos de entrenamiento, validación y prueba.
- Exploración y visualización de imágenes.
- Implementación de generadores de datos con `ImageDataGenerator`.
- Construcción y compilación de una red neuronal basada en EfficientNetB0.
- Entrenamiento del modelo utilizando callbacks.
- Evaluación mediante métricas de clasificación y matriz de confusión.
- Implementación de un script para realizar predicciones individuales.
- Documentación del proceso y de las soluciones técnicas aplicadas.

### 11.2. Resultados obtenidos

| Indicador | Resultado |
|---|---:|
| Imágenes originales | 25.000 |
| Imágenes de entrenamiento | 20.000 |
| Imágenes de validación | 2.500 |
| Imágenes de prueba | 2.500 |
| Arquitectura | EfficientNetB0 |
| Nueva accuracy de validación | Pendiente de reentrenar |
| Accuracy de prueba independiente | Pendiente de reentrenar |
| Modelo guardado | `best_model.keras` |

### 11.3. Limitaciones y posibles mejoras

Aunque el proyecto se completó correctamente, existen algunas oportunidades de mejora:

1. **Pesos preentrenados:** experimentar con EfficientNetB0 utilizando pesos preentrenados de ImageNet mediante transferencia de aprendizaje (*transfer learning*).
2. **Aumento de datos:** incorporar técnicas de *data augmentation* para mejorar la capacidad de generalización del modelo.
3. **Optimización del entrenamiento:** probar diferentes tamaños de batch, optimizadores y tasas de aprendizaje.
4. **Visualización de resultados:** incorporar gráficos de evolución de accuracy y loss, así como una visualización de la matriz de confusión.

### 11.4. Conclusión

El proyecto permitió implementar un flujo completo de clasificación de imágenes mediante técnicas de aprendizaje profundo.

El repositorio ahora separa los datos de entrenamiento, validación y prueba. El 91,32 % reportado anteriormente procedía de un conjunto de prueba reutilizado y no es un resultado independiente; es necesario volver a entrenar para informar métricas válidas de test. También se implementó una funcionalidad para realizar predicciones sobre imágenes individuales.

El desarrollo también permitió trabajar con herramientas de Python, TensorFlow, Keras, procesamiento de imágenes, evaluación de modelos y gestión de proyectos mediante Git y GitHub.

---