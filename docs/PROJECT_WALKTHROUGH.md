# Project Walkthrough: Cats and Dogs Image Classifier

## 1. Project Introduction

### Objective

The objective of this project is to develop an image classification model capable of automatically identifying whether an image contains a cat or a dog.

To achieve this, a convolutional neural network based on **EfficientNetB0** is used. This is a deep learning architecture designed for image recognition tasks.

The model is trained using a labeled image dataset and subsequently evaluated to measure its classification performance. A script is also implemented to allow users to provide an individual image and obtain a prediction.

### Technologies Used

- Python 3.13
- TensorFlow 2.21
- Keras 3.15
- EfficientNetB0
- NumPy
- Matplotlib
- Scikit-learn
- Git and GitHub
- GitHub Codespaces

### General Project Structure

The project is organized into the following directories:

- `data/`: contains the data used to train and evaluate the model.
- `docs/`: contains the project documentation.
- `models/`: stores the trained model.
- `src/`: contains the scripts for data preparation, exploration, training, evaluation, and prediction.

## 2. Development Environment Setup

### Development Environment

The project was developed using **GitHub Codespaces**, a cloud-based development environment that allows developers to work directly within a GitHub repository.

A Python virtual environment (`.venv`) was used to keep the project's dependencies isolated from the rest of the system.

### Main Versions

The following versions were used during development:

| Tool | Version |
|---|---|
| Python | 3.13.18 |
| TensorFlow | 2.21.0 |
| Keras | 3.15.1 |

### Dependencies

The project dependencies are recorded in the `requirements.txt` file.

To install the packages, the following command is used:

```bash
pip install -r requirements.txt
```

### Environment Considerations

Training was performed using CPU because the Codespaces environment did not provide a CUDA-compatible GPU.

Additionally, due to the environment's RAM limitations, it was necessary to adjust the batch size during training. This decision is explained in more detail in the training section.

## 3. Dataset Download and Preparation

### 3.1. Dataset Download

The **Dogs vs. Cats** dataset, provided through Google Cloud Storage, was used to develop the classifier.

Source: [Dogs vs. Cats Dataset](https://storage.googleapis.com/datascience-materials/dogs-vs-cats.zip)

The downloaded file was:

```text
data/dogs-vs-cats.zip
```

After extraction, a directory containing 25,000 JPG images was obtained:

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

The original dataset contains:

- 12,500 cat images.
- 12,500 dog images.

### 3.2. Data Organization

The following script was created to prepare the dataset:

```text
src/prepare_dataset.py
```

The objective was to divide the images into two datasets:

- **Training set:** 80% of the images.
- **Test set:** 20% of the images.

The split was performed separately for each category, maintaining the same proportion of cat and dog images.

A random seed (`random.seed(42)`) was used to make the image selection reproducible.

### 3.3. Final Dataset Distribution

The preparation process produced the following distribution:

| Category | Training | Test | Total |
|---|---:|---:|---:|
| Cats | 10,000 | 2,500 | 12,500 |
| Dogs | 10,000 | 2,500 | 12,500 |
| **Total** | **20,000** | **5,000** | **25,000** |

The final directory structure was organized as follows:

```text
data/
└── processed/
    ├── train/
    │   ├── cats/
    │   └── dogs/
    └── test/
        ├── cats/
        └── dogs/
```

### 3.4. Storage Optimization

The images were organized using `os.link()`, which creates hard links to the original files.

This solution made it possible to separate the images into training and test directories without physically duplicating the files, reducing storage consumption.

### 3.5. Result

The dataset preparation was completed successfully, with 20,000 images allocated to training and 5,000 images allocated to testing.

## 4. Image Exploration and Visualization

### 4.1. Objective

Before training the model, a visual exploration of the dataset was performed to inspect the image contents and become familiar with the data that would be used by the classifier.

### 4.2. Tools Used

The following notebook was created for this stage:

```text
src/explore.ipynb
```

The following libraries were used:

- `os`: to access files and directories.
- `Matplotlib`: to visualize images.
- `TensorFlow Keras`: to load images.

The imports used were:

```python
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
```

### 4.3. Image Visualization

The first nine images from each category were visualized separately:

- Nine cat images.
- Nine dog images.

This made it possible to visually inspect the images and verify that the files were correctly organized by category.

### 4.4. Result

The visual exploration was completed successfully. Both categories were displayed, confirming that the dataset contained cat and dog images suitable for continuing with the classification process.

## 5. Data Loading with ImageDataGenerator

### 5.1. Objective

After preparing the dataset, a mechanism was needed to load the images and supply them to the model during training and evaluation.

For this purpose, `ImageDataGenerator` from TensorFlow Keras was used. It allows images to be loaded progressively in batches rather than loading all 25,000 images into memory simultaneously.

### 5.2. Configuration

The following file was created:

```text
src/data_loader.py
```

The following import was used:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

The following parameters were defined:

```python
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8

train_dir = "data/processed/train"
test_dir = "data/processed/test"
```

Images were resized to 224 × 224 pixels to match the input dimensions specified later in the EfficientNetB0 architecture.

The batch size was set to 8 images to reduce memory consumption during processing.

### 5.3. Generator Creation

Two independent generators were created:

- `trdata`: loads the training images.
- `tsdata`: loads the test images.

Both use `ImageDataGenerator()` without additional transformations.

The following settings were applied:

- `class_mode="categorical"` because the model performs a two-category classification using a Softmax output layer.
- `shuffle=True` to shuffle images during training.
- `shuffle=False` to preserve image order during evaluation, allowing predictions to be correctly compared with the actual labels.

### 5.4. Image Loading

The generators use `flow_from_directory()` to automatically locate images and assign their corresponding labels based on the directory structure.

The output confirmed that the images were loaded correctly:

```text
Found 20000 images belonging to 2 classes.
Found 5000 images belonging to 2 classes.
```

### 5.5. Preprocessing Considerations

No `rescale=1./255` transformation was added to the generator because EfficientNetB0 incorporates its own input rescaling and preprocessing.

This configuration was maintained to remain compatible with the selected architecture.

### 5.6. Result

Data loading was completed successfully:

- 20,000 images available for training.
- 5,000 images available for evaluation.
- Images resized to 224 × 224 pixels.
- Batch processing using 8 images per batch.

The `trdata` and `tsdata` generators were ready to supply data to the model during the following stages.

## 6. EfficientNetB0 Model Architecture

### 6.1. Architecture Selection

**EfficientNetB0** was selected to develop the classifier. It is a convolutional neural network architecture designed for image recognition and classification tasks.

The implementation available through `keras.applications` was used.

The architecture was configured with `include_top=False`, allowing the convolutional base of EfficientNetB0 to be used without its original classification layer.

Additionally, `weights=None` was specified, meaning that the model starts training from scratch without using pretrained weights.

### 6.2. Model Implementation

The following file was created:

```text
src/model.py
```

The model was built using a Sequential architecture with the following layers:

1. **EfficientNetB0:** extracts relevant features from the images.
2. **GlobalAveragePooling2D:** reduces the spatial dimensions of the extracted features.
3. **Dense (128 neurons, ReLU):** processes the features for classification.
4. **Dense (2 neurons, Softmax):** produces the probabilities corresponding to the two categories.

### 6.3. Input Configuration

The input dimensions were defined as:

```python
input_shape=(224, 224, 3)
```

This corresponds to images measuring 224 × 224 pixels with three color channels (RGB).

### 6.4. Model Compilation

The following parameters were used to configure the learning process:

| Parameter | Configuration |
|---|---|
| Optimizer | Adam |
| Loss function | Categorical Crossentropy |
| Metric | Accuracy |
| Hidden layer activation | ReLU |
| Output layer activation | Softmax |

`categorical_crossentropy` was selected because the labels are provided in categorical format through the data generators.

### 6.5. Architecture Summary

After executing `model.summary()`, the following architecture was displayed:

| Layer | Output | Parameters |
|---|---|---:|
| EfficientNetB0 | (None, 7, 7, 1280) | 4,049,571 |
| GlobalAveragePooling2D | (None, 1280) | 0 |
| Dense | (None, 128) | 163,968 |
| Dense | (None, 2) | 258 |

**Total parameters:** 4,213,797.

Of these, 4,171,774 are trainable and 42,023 are non-trainable.

### 6.6. Result

The architecture was successfully built and compiled. The model was ready to learn how to distinguish between cat and dog images during the training stage.

## 7. Model Training

### 7.1. Training Setup

The following file was created to train the neural network:

```text
src/train.py
```

The data generators (`trdata` and `tsdata`), the previously defined model, and two Keras callbacks were imported:

- `ModelCheckpoint`
- `EarlyStopping`

### 7.2. ModelCheckpoint Configuration

`ModelCheckpoint` was used to automatically save the best model during training.

The configuration was:

| Parameter | Value |
|---|---|
| File | `models/best_model.keras` |
| Monitor | `val_accuracy` |
| `save_best_only` | `True` |
| Mode | `max` |

This configuration ensures that the file stores the weights corresponding to the epoch with the highest validation accuracy.

### 7.3. EarlyStopping Configuration

`EarlyStopping` was also implemented to monitor the training process and avoid continuing unnecessarily when validation loss stops improving.

Its configuration was:

| Parameter | Value |
|---|---|
| Monitor | `val_loss` |
| Patience | 3 |
| `restore_best_weights` | `True` |

At the end of training, the callback restored the weights corresponding to the epoch with the lowest validation loss.

### 7.4. Training Parameters

The model was trained using `model.fit()` with the following parameters:

- Training generator: `trdata`.
- Validation data: `tsdata`.
- Maximum number of epochs: 10.
- Batch size: 8 images.
- Callbacks: `ModelCheckpoint` and `EarlyStopping`.

### 7.5. Memory Issues and Adjustments

During the initial tests, a batch size of 32 images was used. However, the training process was terminated by the environment, most likely due to the available RAM limitations in Codespaces.

To solve this issue, the batch size was reduced to 8 images.

With this adjustment, training was completed successfully.

Additionally, the environment did not provide a CUDA-compatible GPU, so training was performed using CPU. Each epoch took approximately 44 to 49 minutes.

### 7.6. Training Results

The model was trained for 10 epochs. The results were:

| Epoch | Accuracy | Loss | Val. Accuracy | Val. Loss |
|---:|---:|---:|---:|---:|
| 1 | 59.15% | 0.6762 | 58.26% | 0.6691 |
| 2 | 68.83% | 0.5900 | 72.60% | 0.5391 |
| 3 | 77.57% | 0.4809 | 82.28% | 0.3982 |
| 4 | 82.82% | 0.3889 | 84.68% | 0.3535 |
| 5 | 86.25% | 0.3210 | 88.12% | 0.2820 |
| 6 | 89.47% | 0.2581 | 88.32% | 0.2881 |
| 7 | 91.17% | 0.2183 | 90.34% | 0.2397 |
| 8 | 92.56% | 0.1858 | 91.10% | 0.2306 |
| 9 | 93.67% | 0.1603 | 91.32% | 0.2307 |
| 10 | 94.48% | 0.1395 | 91.12% | 0.2366 |

### 7.7. Best Model Selection

The `ModelCheckpoint` callback, which monitored `val_accuracy`, saved the model corresponding to epoch 9, which achieved the highest validation accuracy: **91.32%**.

On the other hand, `EarlyStopping` monitored `val_loss` and restored the weights from epoch 8, which achieved the lowest validation loss: 0.2306.

It is important to distinguish between these two results: the `models/best_model.keras` file contains the model saved by `ModelCheckpoint`, corresponding to epoch 9.

### 7.8. Validation Data Considerations

During training, `tsdata` was used as the validation dataset through the `validation_data` parameter.

The same dataset was subsequently used for the final model evaluation.

Therefore, although an accuracy of 91.32% was achieved, the evaluation dataset does not constitute an independent test set because its results were also used to monitor training.

This limitation should be considered when interpreting the results.

### 7.9. Result

Training was completed successfully, generating the following file:

```text
models/best_model.keras
```

This file contains the model selected by `ModelCheckpoint` and is used in the subsequent evaluation and individual prediction stages.

## 8. Model Evaluation

### 8.1. Objective

Once training was completed, the model was evaluated to measure its classification performance and determine how accurately it identified cat and dog images.

For this purpose, the model saved through `ModelCheckpoint` and the `tsdata` generator were used.

### 8.2. Implementation

The following file was created:

```text
src/evaluate.py
```

The following tools were used in this script:

- `load_model`: loads the trained model.
- `model.evaluate()`: calculates loss and accuracy.
- `model.predict()`: generates predictions for the images.
- `numpy.argmax()`: converts prediction probabilities into class labels.
- `accuracy_score`: calculates accuracy.
- `confusion_matrix`: generates the confusion matrix.
- `classification_report`: provides classification metrics.

### 8.3. General Metrics

The results obtained were:

| Metric | Result |
|---|---:|
| Test Loss | 0.2307 |
| Test Accuracy | 91.32% |
| Accuracy Score | 91.32% |

The model correctly classified 4,566 out of the 5,000 evaluated images.

### 8.4. Confusion Matrix

The resulting confusion matrix was:

| Actual Category | Predicted: Cats | Predicted: Dogs |
|---|---:|---:|
| Cats | 2,184 | 316 |
| Dogs | 118 | 2,382 |

Interpretation:

- **2,184 cats** were correctly classified.
- **316 cats** were incorrectly classified as dogs.
- **2,382 dogs** were correctly classified.
- **118 dogs** were incorrectly classified as cats.

In total, the model made 4,566 correct predictions and 434 incorrect predictions.

### 8.5. Classification Report

The metrics obtained for each category were:

| Category | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Cats | 0.95 | 0.87 | 0.91 | 2,500 |
| Dogs | 0.88 | 0.95 | 0.92 | 2,500 |
| **Macro avg** | **0.92** | **0.91** | **0.91** | **5,000** |
| **Weighted avg** | **0.92** | **0.91** | **0.91** | **5,000** |

### 8.6. Results Interpretation

The results show that the model achieved an overall accuracy of 91.32%.

The following observations can be made:

- The `cats` category achieved 95% precision and 87% recall.
- The `dogs` category achieved 88% precision and 95% recall.

This indicates that the model correctly identified a higher proportion of dogs than cats, although it produced more false positives when classifying cats as dogs.

### 8.7. Result

The evaluation confirmed that the model can distinguish between both categories, achieving an overall accuracy of 91.32% on the dataset used.

**Note:** These results should be interpreted considering that the test dataset was also used as the validation dataset during training, as explained in the previous section.

## 9. Individual Predictions

### 9.1. Objective

In addition to evaluating the model using the test dataset, a feature was implemented to allow users to provide an individual image and obtain a prediction.

This makes it possible to test the classifier using manually selected images.

### 9.2. Implementation

The following file was created:

```text
src/predict.py
```

The script performs the following operations:

1. Loads the trained model from `models/best_model.keras`.
2. Asks the user to provide an image path.
3. Loads the image using `load_img()`.
4. Resizes the image to 224 × 224 pixels.
5. Converts the image into a numerical array using `img_to_array()`.
6. Adds an additional dimension to create a batch containing one image using `numpy.expand_dims()`.
7. Runs `model.predict()` to obtain the classification probabilities.
8. Uses `numpy.argmax()` to select the category with the highest probability.
9. Displays the predicted category and confidence percentage.

### 9.3. Execution

From the project root directory, the script can be executed using:

```bash
python src/predict.py
```

The program requests the image path:

```text
Enter the image path:
```

The user enters the corresponding path, and the model returns the prediction.

### 9.4. Tests Performed

Two images from the original dataset were used for testing:

| Image | Predicted Category | Confidence |
|---|---|---:|
| `cat.10.jpg` | Cats | 98.88% |
| `dog.10.jpg` | Dogs | 99.26% |

Both images were classified correctly.

### 9.5. Result

The individual prediction functionality was successfully implemented, allowing the trained model to classify cat and dog images independently.

The displayed percentage represents the probability assigned by the model to the predicted category, not a guarantee that the classification is correct.

## 10. Problems Encountered and Solutions

During project development, several challenges arose related to the execution environment, dependencies, and available resources.

### 10.1. RAM Limitations

**Problem:**

During the initial training tests, a batch size of 32 images was used. The process was terminated by the Codespaces environment, most likely due to memory consumption.

**Solution:**

The batch size was reduced to 8 images to decrease memory usage and allow training to complete successfully.

### 10.2. ImageDataGenerator Compatibility with Keras

**Problem:**

Importing `ImageDataGenerator` from `keras.preprocessing.image` was not supported by the installed Keras version.

**Solution:**

The import was changed to use TensorFlow Keras:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

This allowed the training and evaluation generators to be created successfully.

### 10.3. Use of fit_generator()

**Problem:**

During development, it was identified that `fit_generator()` was no longer necessary in the installed Keras version.

**Solution:**

`model.fit()` was used directly with the image generators, as it supports this type of input.

### 10.4. Training Without a GPU

**Problem:**

The GitHub Codespaces environment used for this project did not provide a compatible CUDA GPU.

**Solution:**

Training was performed using CPU. Although this increased processing time, it allowed all 10 epochs to be completed without changing the infrastructure.

### 10.5. Dataset Storage Optimization

**Problem:**

Organizing the dataset required separating the original images into training and test directories, which could lead to file duplication and unnecessary storage consumption.

**Solution:**

`os.link()` was used to create hard links to the original images, avoiding physical file duplication.

### 10.6. File and Dependency Management

A Python virtual environment was used to isolate project dependencies, and a `requirements.txt` file was maintained to record the required packages.

The `.gitignore` file was also configured to exclude datasets, ZIP files, and trained models from version control.

This helped keep the repository lightweight and prevented large files from being uploaded to GitHub.

## 11. Final Results and Conclusions

### 11.1. Completed Objectives

A binary image classifier capable of distinguishing between cats and dogs was successfully developed using EfficientNetB0.

The following tasks were completed:

- Downloading and organizing a dataset containing 25,000 images.
- Splitting the data into training and test sets.
- Exploring and visualizing images.
- Implementing data generators using `ImageDataGenerator`.
- Building and compiling an EfficientNetB0-based neural network.
- Training the model using callbacks.
- Evaluating the model using classification metrics and a confusion matrix.
- Implementing a script for individual image predictions.
- Documenting the development process and technical solutions.

### 11.2. Results Summary

| Indicator | Result |
|---|---:|
| Original images | 25,000 |
| Training images | 20,000 |
| Test images | 5,000 |
| Architecture | EfficientNetB0 |
| Epochs completed | 10 |
| Best validation accuracy | 91.32% |
| Reported test accuracy | 91.32% |
| Saved model | `best_model.keras` |

### 11.3. Limitations and Possible Improvements

Although the project was completed successfully, there are several opportunities for improvement:

1. **Independent test set:** reserve an independent validation set for monitoring training and keep the test set exclusively for final evaluation.
2. **Pretrained weights:** experiment with EfficientNetB0 using ImageNet pretrained weights through transfer learning.
3. **Data augmentation:** incorporate data augmentation techniques to improve the model's generalization ability.
4. **Training optimization:** experiment with different batch sizes, optimizers, and learning rates.
5. **Results visualization:** incorporate training and validation accuracy/loss plots, as well as a visualization of the confusion matrix.

### 11.4. Conclusion

This project made it possible to implement a complete image classification workflow using deep learning techniques.

An EfficientNetB0 model was successfully trained and evaluated, achieving a reported accuracy of 91.32%. An additional feature was also implemented to perform predictions on individual images.

The development process provided practical experience with Python, TensorFlow, Keras, image processing, model evaluation, and project management using Git and GitHub.

---