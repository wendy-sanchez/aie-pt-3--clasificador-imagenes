# Dog vs. Cat Image Classification

**An image classification project using Deep Learning and EfficientNetB0 to distinguish between cats and dogs.**

[English](README.md) | [Español](README.es.md)

---

## Overview

This project implements a binary image classification system capable of identifying whether an image contains a cat or a dog.

The model is built using **EfficientNetB0**, a convolutional neural network architecture, with TensorFlow and Keras. The system is trained on the Dogs vs. Cats dataset and evaluated using an independent test dataset.

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

The original dataset is divided into three independent subsets using an 80/10/10 split:

| Dataset | Images | Percentage |
| :--- | ---: | ---: |
| Training | 20,000 | 80% |
| Validation | 2,500 | 10% |
| Testing | 2,500 | 10% |
| **Total** | **25,000** | **100%** |

Each subset contains images from both classes, with an equal distribution of cats and dogs.

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

Images are loaded progressively using Keras `ImageDataGenerator` and `flow_from_directory()`.

This approach avoids loading the entire dataset into memory.

The dataset is divided into:

- `train/`: images used to update model weights.
- `validation/`: images used for model selection and early stopping.
- `test/`: images reserved exclusively for final evaluation.

Each directory contains two class folders: `cats/` and `dogs/`.

Images are resized to 224 × 224 pixels when loaded by the generators.

The dataset preparation script creates a reproducible split using a fixed random seed. It uses hard links to avoid duplicating image files.

## Training and Optimization

The model is trained using the following configuration:

- Batch size: 8.
- Maximum epochs: 10.
- Optimizer: Adam.
- Loss function: Categorical Crossentropy.
- Evaluation metric: Accuracy.

Two callbacks are implemented:

- **ModelCheckpoint:** saves the model with the best validation accuracy.
- **EarlyStopping:** monitors validation loss, stops training when it stops improving, and restores the weights corresponding to the best validation loss.

The best model according to validation accuracy is saved as:

`models/best_model.keras`

Training completed all 10 epochs. The highest validation accuracy was achieved at epoch 8.

## Model Evaluation

The final evaluation is performed using the independent test subset, which is not used during training, early stopping, or model checkpoint selection.

The following metrics are calculated using Scikit-learn:

- Accuracy.
- Loss.
- Confusion Matrix.
- Precision.
- Recall.
- F1-score.

### Final Results

The model achieved the following results on the 2,500 test images:

| Metric | Result |
| :--- | ---: |
| Test Accuracy | **88.76%** |
| Test Loss | 0.2581 |
| Correct Predictions | 2,219 |
| Incorrect Predictions | 281 |

The best validation accuracy achieved during training was **91.48%**, at epoch 8.

### Confusion Matrix

The confusion matrix shows the relationship between the actual and predicted classes:

| Actual Class | Predicted Cats | Predicted Dogs |
| :--- | ---: | ---: |
| Cats | 1,061 | 189 |
| Dogs | 92 | 1,158 |

### Classification Report

| Class | Precision | Recall | F1-score | Support |
| :--- | ---: | ---: | ---: | ---: |
| Cats | 0.92 | 0.85 | 0.88 | 1,250 |
| Dogs | 0.86 | 0.93 | 0.89 | 1,250 |
| **Macro Average** | **0.89** | **0.89** | **0.89** | **2,500** |
| **Weighted Average** | **0.89** | **0.89** | **0.89** | **2,500** |

The model achieved a balanced F1-score across both classes. It showed higher recall for dogs, while precision was higher for cats.

These results provide an independent measure of the model's generalization performance.

## Individual Image Predictions

The project includes a script to classify individual images using the trained model.

Run:

```bash
python src/predict.py
```

Enter the path to an image when prompted.

**Example using an image from the test dataset:**

```text
Enter the image path: data/processed_split/test/cats/cat.1.jpg
Predicted class: cats
Confidence: 99.95%
```

The confidence score represents the probability assigned by the model to its predicted class. It does not guarantee that the prediction is correct.

## Project Structure

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

This creates separate training, validation, and testing directories under `data/processed_split/`. The existing `data/processed/` directory, if present, is left untouched.

### 5. Explore the dataset

Open the notebook:

```text
src/explore.ipynb
```

### 6. Train the model

```bash
python src/train.py
```

The best model according to validation accuracy will be saved in:

```text
models/best_model.keras
```

### 7. Evaluate the model

```bash
python src/evaluate.py
```

This evaluates the saved model using the independent test subset and displays the evaluation metrics, confusion matrix, and classification report.

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
- The test subset must remain independent and should only be used for final model evaluation.

---

**Developed by Wendy Sanchez**  
**Machine Learning & Deep Learning Project**