# Clasificador de Imagen Dog vs. Cat

Un proyecto de clasificación de imágenes que utiliza aprendizaje profundo (Deep Learning) y EfficientNetB0 para distinguir entre gatos y perros.

[English](README.md) | [Español](README.es.md)

---

## Descripción general

Este proyecto implementa un sistema de clasificación binaria de imágenes capaz de identificar si una fotografía contiene un gato o un perro.

El modelo está desarrollado utilizando **EfficientNetB0**, una arquitectura de red neuronal convolucional, junto con TensorFlow y Keras. El sistema se entrena con el conjunto de datos Dogs vs. Cats y se evalúa utilizando un directorio independiente de imágenes de prueba.

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

Las imágenes se organizan en directorios de entrenamiento, validación y prueba, con subdirectorios independientes para cada clase.

| Conjunto | Imágenes |
|---|---:|
| Entrenamiento | 20.000 |
| Validación | 2.500 |
| Prueba | 2.500 |
| **Total** | **25.000** |

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

Debido a que el entorno de trabajo dispone de menos de 12 GB de RAM, las imágenes se cargan progresivamente utilizando `ImageDataGenerator` y `flow_from_directory()` de Keras.

Este procedimiento evita cargar todo el conjunto de datos simultáneamente en memoria.

El dataset se divide en:

- `train/`: imágenes utilizadas para actualizar los pesos del modelo.
- `validation/`: imágenes utilizadas para decidir la parada temprana y seleccionar el mejor modelo.
- `test/`: imágenes reservadas exclusivamente para la evaluación final.

Cada directorio contiene dos subcarpetas correspondientes a las clases: `cats/` y `dogs/`.

Las imágenes se redimensionan a 224 × 224 píxeles durante su carga mediante los generadores.

## Entrenamiento y optimización

El modelo se entrena con la siguiente configuración:

- Batch size: 8.
- Épocas: 10.
- Optimizador: Adam.
- Función de pérdida: Categorical Crossentropy.
- Métrica de evaluación: Accuracy.

Se implementan dos funciones callback:

- **ModelCheckpoint:** guarda el modelo que obtiene la mejor accuracy de validación.
- **EarlyStopping:** detiene el entrenamiento cuando la pérdida de validación deja de mejorar y restaura los pesos de la mejor época.

El mejor modelo se guarda en:

`models/best_model.keras`

## Evaluación del modelo

El modelo se evalúa utilizando las siguientes métricas:

- Accuracy.
- Loss.
- Matriz de confusión.
- Precision.
- Recall.
- F1-score.

Se utiliza Scikit-learn para calcular las métricas de clasificación.

### Resultados

La accuracy del 91,32 % reportada anteriormente se midió sobre imágenes que también se utilizaron para validación y selección del modelo. No es un resultado de prueba independiente y no se presenta como tal. Ejecuta el entrenamiento y luego `src/evaluate.py` para obtener las métricas del conjunto de prueba reservado.

## Predicciones de imágenes individuales

El proyecto incluye un script de predicción que carga el modelo entrenado y clasifica imágenes individuales.

El programa devuelve:

- Clase predicha (gatos o perros).
- Nivel de confianza de la predicción.

Ejemplo:

```text
Predicted class: cats
Confidence: 98.88%
```

## Estructura del proyecto

```text
aie-pt-3--clasificador-imagenes/
│
├── data/
│   ├── raw/
│   ├── processed_split/
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

El mejor modelo se guardará en:

```text
models/best_model.keras
```

### 7. Evaluar el modelo

```bash
python src/evaluate.py
```

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

---

**Developed by Wendy Sanchez**  
**Machine Learning & Deep Learning Project**