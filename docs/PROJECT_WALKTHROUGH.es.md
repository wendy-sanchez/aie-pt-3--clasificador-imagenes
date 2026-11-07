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

Se crearon tres generadores independientes:

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

El proceso de entrenamiento utiliza el conjunto de entrenamiento para actualizar los pesos del modelo y el conjunto de validación para monitorizar su rendimiento y seleccionar el mejor modelo.

El conjunto de prueba (`tsdata`) no se utiliza durante el entrenamiento ni para seleccionar el modelo. Se reserva exclusivamente para la evaluación final.

### 7.2. Configuración de ModelCheckpoint

Se utilizó `ModelCheckpoint` para guardar automáticamente el mejor modelo durante el entrenamiento.

La configuración fue:

| Parámetro        | Valor                     |
| :--------------- | :------------------------ |
| Archivo          | `models/best_model.keras` |
| Monitorización   | `val_accuracy`            |
| `save_best_only` | `True`                    |
| Modo             | `max`                     |

De esta manera, el archivo conserva los pesos correspondientes a la época con mayor precisión de validación.

### 7.3. Configuración de EarlyStopping

También se implementó `EarlyStopping` para controlar el proceso de entrenamiento y evitar continuar innecesariamente cuando la pérdida de validación deja de mejorar.

Su configuración fue:

| Parámetro              | Valor      |
| :--------------------- | :--------- |
| Monitorización         | `val_loss` |
| Patience               | 3          |
| `restore_best_weights` | `True`     |

Al finalizar, este callback restaura los pesos correspondientes a la época con menor pérdida de validación.

### 7.4. Parámetros de entrenamiento

Se utilizó `model.fit()` para entrenar el modelo con los siguientes parámetros:

- Generador de entrenamiento: `trdata`.
- Datos de validación: `valdata`.
- Número máximo de épocas: 10.
- Batch size: 8 imágenes.
- Callbacks: `ModelCheckpoint` y `EarlyStopping`.

### 7.5. Problemas de recursos y ajustes realizados

Durante las primeras pruebas se utilizó un batch size de 32 imágenes. Sin embargo, el proceso de entrenamiento fue interrumpido por el entorno, posiblemente debido a las limitaciones de recursos disponibles en Codespaces. No se pudo confirmar la causa exacta.

Para solucionar el problema, se redujo el batch size a 8 imágenes.

Con este ajuste, el entrenamiento pudo completarse correctamente.

Además, el entorno no disponía de GPU CUDA, por lo que el entrenamiento se realizó utilizando CPU. Cada época tardó aproximadamente entre 44 y 49 minutos.

### 7.6. Resultados del entrenamiento

Después de corregir el problema de data leakage e implementar conjuntos independientes de entrenamiento, validación y prueba, se volvió a entrenar el modelo.

Los resultados obtenidos fueron los siguientes:

| Época | Accuracy | Loss   | Val. Accuracy | Val. Loss |
| ----: | -------: | -----: | ------------: | --------: |
| 1     | 55,69 %  | 0,6888 | 58,52 %       | 0,6594    |
| 2     | 64,80 %  | 0,6260 | 72,24 %       | 0,5523    |
| 3     | 73,91 %  | 0,5299 | 80,60 %       | 0,4406    |
| 4     | 79,65 %  | 0,4476 | 82,56 %       | 0,4038    |
| 5     | 83,03 %  | 0,3883 | 85,84 %       | 0,3211    |
| 6     | 85,79 %  | 0,3313 | 88,20 %       | 0,2810    |
| 7     | 88,68 %  | 0,2771 | 88,56 %       | 0,2719    |
| 8     | 90,85 %  | 0,2322 | 91,48 %       | 0,2145    |
| 9     | 92,36 %  | 0,1998 | 91,04 %       | 0,2279    |
| 10    | 93,22 %  | 0,1720 | 91,04 %       | 0,2142    |

El modelo mostró una mejora continua en la precisión de entrenamiento durante las 10 épocas.

La mayor precisión de validación se alcanzó en la época 8, con un resultado del 91,48 %.

### 7.7. Selección del mejor modelo

Los dos callbacks monitorizaron métricas de validación diferentes:

- `ModelCheckpoint` monitorizó `val_accuracy` y guardó el modelo de la época 8, que alcanzó la mayor precisión de validación: 91,48 %.
- `EarlyStopping` monitorizó `val_loss` y restauró los pesos de la época 10, que obtuvo la menor pérdida de validación: 0,2142.

Por tanto, el archivo `models/best_model.keras` contiene el modelo seleccionado por `ModelCheckpoint`, correspondiente a la época 8.

Este es el modelo que cargan posteriormente los scripts de evaluación y predicción individual.

### 7.8. Consideraciones sobre los datos de validación

El flujo corregido utiliza `valdata` mediante el parámetro `validation_data`.

El conjunto de validación permite monitorizar el rendimiento del modelo durante el entrenamiento y determinar qué pesos deben guardarse.

El conjunto de prueba (`tsdata`) no se utiliza durante el entrenamiento, la validación, el early stopping ni la selección del checkpoint. Se mantiene reservado para la evaluación final.

Esta separación evita el problema de data leakage identificado durante el proceso de entrenamiento anterior y permite obtener una medición independiente de la capacidad de generalización del modelo.

### 7.9. Resultado

El entrenamiento se completó correctamente después de implementar la división corregida de los datos y reducir el batch size a 8 imágenes.

El proceso generó el siguiente archivo:

```text
models/best_model.keras
```

El modelo guardado corresponde a la época 8, con una precisión de validación del 91,48 %.

Posteriormente, este modelo se utilizó para realizar la evaluación final sobre el conjunto de prueba independiente.

## 8. Evaluación del modelo

### 8.1 Evaluación sobre el conjunto de test

Una vez finalizado el entrenamiento, se evaluó el modelo guardado en `models/best_model.keras` utilizando el conjunto de test (`tsdata`).

Este conjunto contiene 2.500 imágenes que no participaron en el entrenamiento ni en la selección del mejor modelo. Por tanto, permite medir el rendimiento del modelo sobre datos independientes.

Resultados obtenidos:

| Métrica | Resultado |
|---|---:|
| Test Loss | 0.2581 |
| Test Accuracy | 88.76% |
| Predicciones correctas | 2.219 |
| Predicciones incorrectas | 281 |
| Total de imágenes | 2.500 |

La precisión global del 88.76% indica que el modelo clasificó correctamente 2.219 de las 2.500 imágenes evaluadas.

### 8.2 Matriz de confusión

La matriz de confusión obtenida fue:

```text
[[1061  189]
 [  92 1158]]
```

Las filas representan las clases reales y las columnas las clases predichas, en el siguiente orden: `cats` y `dogs`.

| Clase real | Predichas como cats | Predichas como dogs |
|---|---:|---:|
| cats | 1061 | 189 |
| dogs | 92 | 1158 |

Interpretación:

- **Cats:** 1.061 imágenes de gatos fueron clasificadas correctamente, mientras que 189 fueron clasificadas como perros.
- **Dogs:** 1.158 imágenes de perros fueron clasificadas correctamente, mientras que 92 fueron clasificadas como gatos.

El modelo clasificó correctamente más imágenes de perros que de gatos.

### 8.3 Classification Report

El informe de clasificación obtenido fue:

```text
              precision    recall  f1-score   support

        cats       0.92      0.85      0.88      1250
        dogs       0.86      0.93      0.89      1250

    accuracy                           0.89      2500
   macro avg       0.89      0.89      0.89      2500
weighted avg       0.89      0.89      0.89      2500
```

Interpretación de las métricas:

- **Precision:** indica qué proporción de las predicciones de una clase fueron correctas.
- **Recall:** indica qué proporción de las imágenes reales de una clase fueron identificadas correctamente.
- **F1-score:** combina precision y recall en una única métrica.
- **Support:** representa el número de imágenes reales de cada clase evaluadas.

El modelo obtuvo un recall del 93% para perros y del 85% para gatos. Esto indica que identificó una proporción mayor de imágenes de perros que de gatos.

### 8.4 Comparación entre validation y test

La mejor precisión obtenida durante el entrenamiento fue del 91.48% en el conjunto de validación (epoch 8).

Al evaluar el modelo guardado sobre el conjunto de test, la precisión fue del 88.76%.

La diferencia entre ambas métricas es de 2.72 puntos porcentuales. Esta diferencia es compatible con una pérdida moderada de rendimiento al evaluar el modelo sobre imágenes independientes que no participaron en el entrenamiento ni en la selección del checkpoint.

### 8.5 Conclusión de la evaluación

El modelo alcanzó una precisión del 88.76% sobre el conjunto de test, clasificando correctamente 2.219 de las 2.500 imágenes.

Los resultados muestran que EfficientNetB0 puede distinguir ambas categorías con un rendimiento satisfactorio, aunque todavía presenta errores de clasificación, especialmente al identificar algunas imágenes de gatos como perros.

La evaluación se realizó utilizando un conjunto de test independiente, evitando reutilizar los datos de entrenamiento o validación para medir el rendimiento final.

## 9. Predicciones individuales

### 9.1 Objetivo

Además de la evaluación general del modelo, se realizaron pruebas individuales para comprobar cómo clasifica imágenes concretas.

Para ello se utilizó el script `src/predict.py`, que permite introducir la ruta de una imagen y obtener:

- La clase predicha (`cats` o `dogs`).
- El nivel de confianza de la predicción.

### 9.2 Primera prueba: imagen del conjunto de entrenamiento

Se utilizó la siguiente imagen:

```text
data/raw/dogs-vs-cats/train/cat.10.jpg
```

Resultado obtenido:

```text
Predicted class: cats
Confidence: 99.22%
```

El modelo identificó correctamente la imagen como un gato, con una confianza del 99.22%.

Esta imagen pertenece al conjunto de entrenamiento, por lo que no constituye una prueba independiente del rendimiento del modelo.

### 9.3 Segunda prueba: imagen del conjunto de test

Se utilizó una imagen del conjunto de test:

```text
data/processed_split/test/cats/cat.1.jpg
```

Resultado obtenido:

```text
Predicted class: cats
Confidence: 99.95%
```

El modelo identificó correctamente la imagen como un gato, con una confianza del 99.95%.

A diferencia de la primera prueba, esta imagen pertenece al conjunto de test, que no se utilizó durante el entrenamiento ni para seleccionar el mejor checkpoint.

### 9.4 Interpretación de los resultados

Ambas predicciones individuales fueron correctas.

La confianza representa la probabilidad que el modelo asigna a la clase predicha mediante su capa de salida Softmax. No debe interpretarse como una garantía de que la clasificación sea correcta.

Estas pruebas permiten comprobar el funcionamiento del script de predicción con imágenes concretas. Sin embargo, el rendimiento general del modelo debe valorarse principalmente mediante la evaluación completa del conjunto de test, cuyos resultados se presentan en la sección anterior.

## 10. Problemas encontrados y soluciones

Durante el desarrollo del proyecto surgieron diferentes dificultades relacionadas con el entorno de ejecución, las dependencias y los recursos disponibles.

### 10.1. Limitaciones de memoria RAM

**Problema:**

Durante las primeras pruebas, el proceso de entrenamiento se interrumpió antes de completar la primera época. El problema pudo estar relacionado con las limitaciones de recursos disponibles en GitHub Codespaces, aunque no se pudo confirmar la causa exacta.

**Solución:**

Para solucionar el problema, se redujo el tamaño del batch de 32 a 8 imágenes. Con esta configuración, el entrenamiento pudo completarse correctamente.

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

La organización del dataset requería separar las imágenes originales en directorios de entrenamiento, validación y prueba, lo que podía generar duplicación de archivos y un consumo innecesario de almacenamiento.

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
- División de los datos en conjuntos independientes de entrenamiento, validación y prueba.
- Exploración y visualización de imágenes.
- Implementación de generadores de datos con `ImageDataGenerator`.
- Construcción y compilación de una red neuronal basada en EfficientNetB0.
- Entrenamiento del modelo utilizando `ModelCheckpoint` y `EarlyStopping`.
- Evaluación mediante métricas de clasificación y matriz de confusión.
- Implementación de un script para realizar predicciones individuales.
- Documentación del proceso de desarrollo y de las soluciones técnicas aplicadas.

### 11.2. Resumen de resultados

El modelo final se evaluó utilizando un conjunto de prueba independiente compuesto por 2.500 imágenes.

| Indicador | Resultado |
| :--- | ---: |
| Imágenes originales | 25.000 |
| Imágenes de entrenamiento | 20.000 |
| Imágenes de validación | 2.500 |
| Imágenes de prueba | 2.500 |
| Arquitectura | EfficientNetB0 |
| Parámetros totales | 4.213.797 |
| Mejor accuracy de validación | 91.48% |
| Accuracy de prueba independiente | 88.76% |
| Test loss | 0.2581 |
| Predicciones correctas | 2.219 |
| Predicciones incorrectas | 281 |
| Modelo guardado | `best_model.keras` |

El modelo clasificó correctamente 2.219 de las 2.500 imágenes del conjunto de prueba, alcanzando una accuracy del 88.76%.

La matriz de confusión y el informe de clasificación proporcionan información adicional sobre el rendimiento en cada categoría. El modelo obtuvo un recall del 85% para gatos y del 93% para perros.

La mejor accuracy de validación fue del 91.48%, alcanzada durante la época 8. El checkpoint guardado se seleccionó utilizando `val_accuracy`, mientras que `EarlyStopping` monitorizó `val_loss`.

### 11.3. Archivos principales del proyecto

El proyecto está organizado en diferentes scripts, cada uno responsable de una etapa específica del proceso de clasificación:

- `src/prepare_dataset.py`: prepara el dataset y crea las divisiones de entrenamiento, validación y prueba.
- `src/data_loader.py`: carga las imágenes y crea los generadores de datos.
- `src/model.py`: define y compila el modelo basado en EfficientNetB0.
- `src/train.py`: ejecuta el entrenamiento del modelo y aplica los callbacks.
- `src/evaluate.py`: evalúa el modelo guardado utilizando el conjunto de prueba independiente.
- `src/predict.py`: realiza predicciones individuales sobre imágenes.
- `models/best_model.keras`: almacena el mejor checkpoint seleccionado durante el entrenamiento.

### 11.4. Limitaciones y posibles mejoras

Aunque el proyecto se completó correctamente, existen varias oportunidades de mejora:

1. **Pesos preentrenados:** experimentar con EfficientNetB0 utilizando pesos preentrenados de ImageNet mediante transferencia de aprendizaje (*transfer learning*).
2. **Aumento de datos:** incorporar técnicas de *data augmentation* para mejorar la capacidad de generalización del modelo.
3. **Optimización del entrenamiento:** probar diferentes tamaños de batch, optimizadores y tasas de aprendizaje.
4. **Visualización de resultados:** incorporar gráficos de evolución de accuracy y loss, así como una visualización de la matriz de confusión.
5. **Rendimiento del modelo:** analizar los errores de clasificación, especialmente las imágenes de gatos clasificadas incorrectamente como perros, para identificar posibles mejoras.

### 11.5. Conclusión

El proyecto permitió implementar un flujo completo de clasificación de imágenes mediante técnicas de aprendizaje profundo.

El dataset se dividió en conjuntos independientes de entrenamiento, validación y prueba, garantizando que la evaluación final se realizara con datos que no habían participado en el entrenamiento ni en la selección del checkpoint.

El clasificador basado en EfficientNetB0 alcanzó una accuracy del 88.76% sobre el conjunto de prueba independiente. Los resultados demuestran que el modelo puede distinguir entre gatos y perros con un rendimiento satisfactorio, aunque todavía presenta algunos errores de clasificación.

Las pruebas de predicción individual también permitieron comprobar el funcionamiento del script con imágenes concretas.

El desarrollo permitió adquirir experiencia práctica con Python, TensorFlow, Keras, procesamiento de imágenes, entrenamiento y evaluación de modelos, y gestión de proyectos mediante Git y GitHub.
---