# Dog vs. Cat Image Classification

**An image classification project using Deep Learning and EfficientNetB0 to distinguish between cats and dogs.**

[English](README.md) | [Español](README.es.md)

---

## Overview

This project implements a binary image classification system capable of identifying whether an image contains a cat or a dog.

The model is built using **EfficientNetB0**, a convolutional neural network architecture, with TensorFlow and Keras. The system is trained on the Dogs vs. Cats dataset and evaluated using a separate test directory.

The project was developed as part of a Machine Learning and Deep Learning training exercise.

## Objectives

- Explore and visualize image datasets.
- Prepare and organize images for model training.
- Implement a convolutional neural network using EfficientNetB0.
- Train and optimize the model using callbacks.
- Evaluate classification performance using Scikit-learn.
- Generate predictions for individual images.

## Dataset

The project uses the Dogs vs. Cats dataset, originally associated with the Asirra CAPTCHA challenge.

**Dataset source:** [Dogs vs. Cats Dataset](https://storage.googleapis.com/datascience-materials/dogs-vs-cats.zip)

The dataset contains 25,000 labeled images:

- 12,500 cat images.
- 12,500 dog images.

The images are organized into training and testing directories, with separate subdirectories for each class.

| Dataset | Images |
|---|---:|
| Training | 20,000 |
| Testing | 5,000 |
| **Total** | **25,000** |

## Technologies and Libraries

- Python
- TensorFlow
- Keras 3
- EfficientNetB0
- NumPy
- Matplotlib
- Scikit-learn

## Model Architecture

The classification model uses the following architecture:

1. EfficientNetB0 (without its original classification head).
2. GlobalAveragePooling2D.
3. Dense layer with 128 neurons and ReLU activation.
4. Dense output layer with 2 neurons and Softmax activation.

**Input image dimensions:** 224 × 224 × 3.

**Total parameters:** 4,213,797.

The model is compiled using the Adam optimizer and categorical cross-entropy loss, with accuracy as its evaluation metric.

## Data Processing

Since the available environment has less than 12 GB of RAM, images are loaded progressively using Keras `ImageDataGenerator` and `flow_from_directory()`.

This approach avoids loading the entire dataset into memory.

The dataset is divided into:

- `train/`: training images.
- `test/`: testing images.

Each directory contains two class folders: `cats/` and `dogs/`.

Images are resized to 224 × 224 pixels when loaded by the generators.

## Training and Optimization

The model is trained using the following configuration:

- Batch size: 8.
- Epochs: 10.
- Optimizer: Adam.
- Loss function: Categorical Crossentropy.
- Evaluation metric: Accuracy.

Two callbacks are implemented:

- **ModelCheckpoint:** saves the model with the best validation accuracy.
- **EarlyStopping:** stops training when validation loss stops improving and restores the weights from the best epoch.

The best model is saved as:

`models/best_model.keras`

## Model Evaluation

The model is evaluated using:

- Accuracy.
- Loss.
- Confusion Matrix.
- Precision.
- Recall.
- F1-score.

Scikit-learn is used to calculate the classification metrics.

### Results

| Metric | Result |
|---|---:|
| Test Accuracy | 91.32% |
| Test Loss | 0.2307 |

### Classification Report

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Cats | 0.95 | 0.87 | 0.91 |
| Dogs | 0.88 | 0.95 | 0.92 |
| **Macro Average** | **0.92** | **0.91** | **0.91** |

### Confusion Matrix

| Actual / Predicted | Cats | Dogs |
|---|---:|---:|
| Cats | 2,184 | 316 |
| Dogs | 118 | 2,382 |

The model achieved an overall accuracy of 91.32% on the 5,000 images in the test directory.

**Evaluation note:** The test generator was also used as the validation data during training. Therefore, these results should not be interpreted as a fully independent final evaluation.

## Individual Image Predictions

The project includes a prediction script that loads the trained model and classifies individual images.

It returns:

- Predicted class (cats or dogs).
- Prediction confidence.

Example:

```text
Predicted class: cats
Confidence: 98.88%
```

## Project Structure

```text
aie-pt-3--clasificador-imagenes/
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── train/
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

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/wendy-sanchez/aie-pt-3--clasificador-imagenes.git

cd aie-pt-3--clasificador-imagenes
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Linux / macOS:**

```bash
source .venv/bin/activate
```

**Windows:**

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download and prepare the dataset

Download the dataset from the source link and extract it into:

```text
data/raw/dogs-vs-cats/train/
```

Run the dataset preparation script:

```bash
python src/prepare_dataset.py
```

This creates the training and testing directories.

### 5. Explore the dataset

Open the notebook:

```text
src/explore.ipynb
```

### 6. Train the model

```bash
python src/train.py
```

The best model will be saved in:

```text
models/best_model.keras
```

### 7. Evaluate the model

```bash
python src/evaluate.py
```

### 8. Predict an individual image

```bash
python src/predict.py
```

When prompted, enter the image path, for example:

```text
data/raw/dogs-vs-cats/train/cat.10.jpg
```

## Notes

- The dataset and trained model are excluded from Git tracking through `.gitignore`.
- Download and prepare the dataset before running the training, evaluation, or prediction scripts.
- The project uses 224 × 224 pixel images to match the specified EfficientNetB0 input shape.
- Training performance depends on available hardware. CPU-based training may take considerably longer than GPU-based training.

---

**Developed by Wendy Sanchez**  
**Machine Learning & Deep Learning Project**