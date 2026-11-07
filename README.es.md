# Clasificador de Imágenes Dog vs. Cat

Un proyecto de clasificación de imágenes que utiliza aprendizaje profundo (Deep Learning) y EfficientNetB0 para distinguir entre gatos y perros.

[English](README.md) | [Español](README.es.md)

---

## Descripción general

Este proyecto implementa un sistema de clasificación binaria de imágenes capaz de identificar si una fotografía contiene un gato o un perro.

El modelo está desarrollado utilizando **EfficientNetB0**, una arquitectura de red neuronal convolucional, junto con TensorFlow y Keras. El sistema se entrena con el conjunto de datos Dogs vs. Cats y se evalúa utilizando un conjunto de prueba independiente.

El proyecto se desarrolló como parte de un ejercicio práctico de formación en Machine Learning y Deep Learning.

## Objetivos

- Explorar y visualizar conjuntos de datos de imágenes.
- Preparar y organizar imágenes para el entrenamiento del modelo.
- Implementar una red neuronal convolucional utilizando EfficientNetB0.
- Entrenar y optimizar el modelo mediante funciones callback.
- Evaluar el rendimiento de la clasificación utilizando Scikit-learn.
- Generar predicciones para imágenes individuales.

## Conjunto de datos

El proyecto utiliza el conjunto de datos Dogs vs. Cats, asociado originalmente al desafío CAPTCHA Asirra.

**Fuente del dataset:** [Dogs vs. Cats Dataset](https://storage.googleapis.com/datascience-materials/dogs-vs-cats.zip)

El conjunto de datos contiene 25.000 imágenes etiquetadas:

- 12.500 imágenes de gatos.
- 12.500 imágenes de perros.

El dataset se divide en tres subconjuntos independientes mediante una distribución 80/10/10:

| Conjunto | Imágenes | Porcentaje |
| :--- | ---: | ---: |
| Entrenamiento | 20.000 | 80 % |
| Validación | 2.500 | 10 % |
| Prueba | 2.500 | 10 % |
| **Total** | **25.000** | **100 %** |

Cada subconjunto contiene imágenes de ambas clases, con una distribución equilibrada entre gatos y perros.

## Tecnologías y librerías

- Python
- TensorFlow
- Keras 3
- EfficientNetB0
- NumPy
- Matplotlib
- Scikit-learn

## Arquitectura del modelo

El modelo de clasificación utiliza la siguiente arquitectura:

1. EfficientNetB0 (sin su capa de clasificación original).
2. GlobalAveragePooling2D.
3. Capa Dense de 128 neuronas con activación ReLU.
4. Capa de salida Dense de 2 neuronas con activación Softmax.

**Dimensiones de entrada:** 224 × 224 × 3.

**Parámetros totales:** 4.213.797.

El modelo se compila utilizando el optimizador Adam y la función de pérdida Categorical Crossentropy, con accuracy como métrica de evaluación.

## Procesamiento de datos

Las imágenes se cargan progresivamente utilizando `ImageDataGenerator` y `flow_from_directory()` de Keras.

Este procedimiento evita cargar todo el conjunto de datos simultáneamente en memoria.

El dataset se divide en:

- `train/`: imágenes utilizadas para actualizar los pesos del modelo.
- `validation/`: imágenes utilizadas para seleccionar el mejor modelo y controlar la parada temprana.
- `test/`: imágenes reservadas exclusivamente para la evaluación final.

Cada directorio contiene dos subcarpetas correspondientes a las clases: `cats/` y `dogs/`.

Las imágenes se redimensionan a 224 × 224 píxeles durante su carga mediante los generadores.

El script de preparación utiliza una semilla aleatoria fija para generar una distribución reproducible. Además, utiliza enlaces físicos (*hard links*) para evitar duplicar los archivos de imagen.

## Entrenamiento y optimización

El modelo se entrena con la siguiente configuración:

- Batch size: 8.
- Número máximo de épocas: 10.
- Optimizador: Adam.
- Función de pérdida: Categorical Crossentropy.
- Métrica de evaluación: Accuracy.

Se implementan dos funciones callback:

- **ModelCheckpoint:** guarda el modelo que obtiene la mejor accuracy de validación.
- **EarlyStopping:** monitoriza la pérdida de validación, detiene el entrenamiento cuando deja de mejorar y restaura los pesos correspondientes a la mejor época según esta métrica.

El mejor modelo según la accuracy de validación se guarda en:

`models/best_model.keras`

El entrenamiento completó las 10 épocas. La mejor accuracy de validación se alcanzó durante la época 8.

## Evaluación del modelo

La evaluación final se realiza utilizando el subconjunto de prueba independiente, que no interviene en el entrenamiento, la parada temprana ni la selección del modelo.

Se utilizan las siguientes métricas, calculadas mediante Scikit-learn:

- Accuracy.
- Loss.
- Matriz de confusión.
- Precision.
- Recall.
- F1-score.

### Resultados finales

El modelo obtuvo los siguientes resultados sobre las 2.500 imágenes del conjunto de prueba:

| Métrica | Resultado |
| :--- | ---: |
| Accuracy en test | **88,76 %** |
| Test Loss | 0,2581 |
| Predicciones correctas | 2.219 |
| Predicciones incorrectas | 281 |

La mejor accuracy de validación obtenida durante el entrenamiento fue del **91,48 %**, durante la época 8.

### Matriz de confusión

La matriz de confusión muestra la relación entre las clases reales y las predichas:

| Clase real | Predicción: gatos | Predicción: perros |
| :--- | ---: | ---: |
| Gatos | 1.061 | 189 |
| Perros | 92 | 1.158 |

### Informe de clasificación

| Clase | Precision | Recall | F1-score | Support |
| :--- | ---: | ---: | ---: | ---: |
| Gatos | 0,92 | 0,85 | 0,88 | 1.250 |
| Perros | 0,86 | 0,93 | 0,89 | 1.250 |
| **Promedio macro** | **0,89** | **0,89** | **0,89** | **2.500** |
| **Promedio ponderado** | **0,89** | **0,89** | **0,89** | **2.500** |

El modelo obtiene un F1-score equilibrado entre ambas clases. Presenta un mayor recall para los perros, mientras que la precision es superior para los gatos.

Estos resultados proporcionan una medida independiente de la capacidad de generalización del modelo.

## Predicciones de imágenes individuales

El proyecto incluye un script para clasificar imágenes individuales utilizando el modelo entrenado.

Ejecuta:

```bash
python src/predict.py
```

Cuando el programa lo solicite, introduce la ruta de la imagen.

**Ejemplo utilizando una imagen del conjunto de prueba:**

```text
Enter the image path: data/processed_split/test/cats/cat.1.jpg
Predicted class: cats
Confidence: 99.95%
```

El porcentaje de confianza representa la probabilidad que el modelo asigna a la clase predicha. No garantiza que la predicción sea correcta.

## Estructura del proyecto

```text
aie-pt-3--clasificador-imagenes/
│
├── data/
│   ├── raw/
│   └── processed_split/
│       ├── train/
│       │   ├── cats/
│       │   └── dogs/
│       ├── validation/
│       │   ├── cats/
│       │   └── dogs/
│       └── test/
│           ├── cats/
│           └── dogs/
│
├── models/
│   └── best_model.keras
│
├── src/
│   ├── explore.ipynb
│   ├── prepare_dataset.py
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── .gitignore
└── requirements.txt
```

## Instalación y ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/wendy-sanchez/aie-pt-3--clasificador-imagenes.git

cd aie-pt-3--clasificador-imagenes
```

### 2. Crear y activar un entorno virtual

```bash
python -m venv .venv
```

Activarlo:

**Linux / macOS:**

```bash
source .venv/bin/activate
```

**Windows:**

```bash
.venv\Scripts\activate
```

### 3. Instalar las dependencias

```bash
pip install -r requirements.txt
```

### 4. Descargar y preparar el dataset

Descargar el conjunto de datos desde el enlace proporcionado y descomprimirlo en:

```text
data/raw/dogs-vs-cats/train/
```

Ejecutar el script de preparación:

```bash
python src/prepare_dataset.py
```

Esto creará directorios separados de entrenamiento, validación y prueba en `data/processed_split/`. Si existe `data/processed/`, el script lo dejará intacto.

### 5. Explorar el conjunto de datos

Abrir el notebook:

```text
src/explore.ipynb
```

### 6. Entrenar el modelo

```bash
python src/train.py
```

El mejor modelo según la accuracy de validación se guardará en:

```text
models/best_model.keras
```

### 7. Evaluar el modelo

```bash
python src/evaluate.py
```

Este script carga el modelo guardado, lo evalúa utilizando el subconjunto de prueba independiente y muestra las métricas de evaluación, la matriz de confusión y el informe de clasificación.

### 8. Predecir una imagen individual

```bash
python src/predict.py
```

Cuando el programa solicite la ruta de la imagen, introducir, por ejemplo:

```text
data/raw/dogs-vs-cats/train/cat.10.jpg
```

## Consideraciones

- El dataset y el modelo entrenado están excluidos del seguimiento de Git mediante `.gitignore`.
- Es necesario descargar y preparar el dataset antes de ejecutar los scripts de entrenamiento, evaluación o predicción.
- El proyecto utiliza imágenes de 224 × 224 píxeles para mantener la compatibilidad con las dimensiones de entrada especificadas para EfficientNetB0.
- El tiempo de entrenamiento depende del hardware disponible. El entrenamiento mediante CPU puede tardar considerablemente más que utilizando GPU.
- El subconjunto de prueba debe mantenerse independiente y utilizarse exclusivamente para la evaluación final del modelo.

---

**Desarrollado por Wendy Sanchez**  
**Proyecto de Machine Learning & Deep Learning**