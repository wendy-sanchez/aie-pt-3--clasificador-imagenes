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

The objective is to divide the images into three datasets:

- **Training set:** 80% of the images.
- **Validation set:** 10% of the images.
- **Test set:** 10% of the images.

The split was performed separately for each category, maintaining the same proportion of cat and dog images.

A random seed (`random.seed(42)`) was used to make the image selection reproducible.

### 3.3. Final Dataset Distribution

The preparation process produced the following distribution:

| Category | Training | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Cats | 10,000 | 1,250 | 1,250 | 12,500 |
| Dogs | 10,000 | 1,250 | 1,250 | 12,500 |
| **Total** | **20,000** | **2,500** | **2,500** | **25,000** |

The final directory structure was organized as follows:

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

### 3.4. Storage Optimization

The images were organized using `os.link()`, which creates hard links to the original files.

This solution made it possible to separate the images into training and test directories without physically duplicating the files, reducing storage consumption.

### 3.5. Result

The preparation script creates the three splits under `data/processed_split/`. Existing files under the earlier `data/processed/` path are preserved.

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

dataset_dir = "data/processed_split"
train_dir = f"{dataset_dir}/train"
validation_dir = f"{dataset_dir}/validation"
test_dir = f"{dataset_dir}/test"
```

Images were resized to 224 × 224 pixels to match the input dimensions specified later in the EfficientNetB0 architecture.

The batch size was set to 8 images to reduce memory consumption during processing.

### 5.3. Generator Creation

Three independent generators were created:

- `trdata`: loads the training images.
- `valdata`: loads the validation images.
- `tsdata`: loads the test images, reserved for final evaluation.

Both use `ImageDataGenerator()` without additional transformations.

The following settings were applied:

- `class_mode="categorical"` because the model performs a two-category classification using a Softmax output layer.
- `shuffle=True` to shuffle images during training.
- `shuffle=False` to preserve image order during evaluation, allowing predictions to be correctly compared with the actual labels.

### 5.4. Image Loading

The generators use `flow_from_directory()` to automatically locate images and assign their corresponding labels based on the directory structure.

The output confirms the expected split sizes:

```text
Found 20000 images belonging to 2 classes.
Found 2500 images belonging to 2 classes.
Found 2500 images belonging to 2 classes.
```

### 5.5. Preprocessing Considerations

No `rescale=1./255` transformation was added to the generator because EfficientNetB0 incorporates its own input rescaling and preprocessing.

This configuration was maintained to remain compatible with the selected architecture.

### 5.6. Result

Data loading was completed successfully:

- 20,000 images available for training.
- 2,500 images available for validation.
- 2,500 images reserved for final evaluation.
- Images resized to 224 × 224 pixels.
- Batch processing using 8 images per batch.

The `trdata`, `valdata`, and `tsdata` generators were ready for training, model selection, and final evaluation, respectively.

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

The data generators (`trdata` and `valdata`), the previously defined model, and two Keras callbacks were imported:

- `ModelCheckpoint`
- `EarlyStopping`

The training process uses the training dataset to update the model weights and the validation dataset to monitor performance and select the best model.

The test dataset (`tsdata`) is not used during training or model selection. It is reserved exclusively for the final evaluation.

### 7.2. ModelCheckpoint Configuration

`ModelCheckpoint` was used to automatically save the best model during training.

The configuration was:

| Parameter        | Value                     |
| :--------------- | :------------------------ |
| File             | `models/best_model.keras` |
| Monitor          | `val_accuracy`            |
| `save_best_only` | `True`                    |
| Mode             | `max`                     |

This configuration ensures that the saved model corresponds to the epoch with the highest validation accuracy.

### 7.3. EarlyStopping Configuration

`EarlyStopping` was also implemented to monitor the training process and avoid continuing unnecessarily when validation loss stops improving.

Its configuration was:

| Parameter              | Value      |
| :--------------------- | :--------- |
| Monitor                | `val_loss` |
| Patience               | 3          |
| `restore_best_weights` | `True`     |

At the end of training, this callback restores the weights corresponding to the epoch with the lowest validation loss.

### 7.4. Training Parameters

The model was trained using `model.fit()` with the following parameters:

- Training generator: `trdata`.
- Validation data: `valdata`.
- Maximum number of epochs: 10.
- Batch size: 8 images.
- Callbacks: `ModelCheckpoint` and `EarlyStopping`.

### 7.5. Memory Issues and Adjustments

During the initial tests, a batch size of 32 images was used. However, the training process was terminated by the environment, possibly related to resource limitations in the Codespaces environment. The exact cause was not confirmed.

To solve this issue, the batch size was reduced to 8 images.

With this adjustment, training was completed successfully.

Additionally, the environment did not provide a CUDA-compatible GPU, so training was performed using CPU. Each epoch took approximately 44 to 49 minutes.

### 7.6. Training Results

After correcting the data leakage issue and implementing the independent training, validation, and test splits, the model was trained again.

The following results were obtained:

| Epoch | Accuracy | Loss   | Val. Accuracy | Val. Loss |
| ----: | -------: | -----: | ------------: | --------: |
| 1     | 55.69%   | 0.6888 | 58.52%        | 0.6594    |
| 2     | 64.80%   | 0.6260 | 72.24%        | 0.5523    |
| 3     | 73.91%   | 0.5299 | 80.60%        | 0.4406    |
| 4     | 79.65%   | 0.4476 | 82.56%        | 0.4038    |
| 5     | 83.03%   | 0.3883 | 85.84%        | 0.3211    |
| 6     | 85.79%   | 0.3313 | 88.20%        | 0.2810    |
| 7     | 88.68%   | 0.2771 | 88.56%        | 0.2719    |
| 8     | 90.85%   | 0.2322 | 91.48%        | 0.2145    |
| 9     | 92.36%   | 0.1998 | 91.04%        | 0.2279    |
| 10    | 93.22%   | 0.1720 | 91.04%        | 0.2142    |

The model showed continuous improvement in training accuracy throughout the 10 epochs.

The highest validation accuracy was achieved during epoch 8, reaching 91.48%.

### 7.7. Best Model Selection

The two callbacks monitored different validation metrics:

- `ModelCheckpoint` monitored `val_accuracy` and saved the model from epoch 8, which achieved the highest validation accuracy of 91.48%.
- `EarlyStopping` monitored `val_loss` and restored the weights from epoch 10, which achieved the lowest validation loss of 0.2142.

As a result, the file `models/best_model.keras` contains the model selected by `ModelCheckpoint`, corresponding to epoch 8.

This is the model loaded by the evaluation and individual prediction scripts.

### 7.8. Validation Data Considerations

The corrected pipeline uses `valdata` through the `validation_data` parameter.

The validation dataset is used to monitor the model's performance during training and to determine which model weights should be saved.

The test dataset (`tsdata`) is not used for training, validation, early stopping, or checkpoint selection. It remains reserved for the final evaluation.

This separation prevents the data leakage issue identified during the earlier training process and provides an independent measurement of the model's generalization performance.

### 7.9. Result

Training was completed successfully after implementing the corrected data split and reducing the batch size to 8 images.

The training process generated the following model file:

```text
models/best_model.keras
```

The saved model corresponds to epoch 8, with a validation accuracy of 91.48%.

This model was subsequently used for the final evaluation on the independent test dataset.

## 8. Model Evaluation

### 8.1. Objective

Once training was completed, the model was evaluated to measure its classification performance and determine how accurately it identified cat and dog images.

For this purpose, the model saved through `ModelCheckpoint` and the independent test dataset (`tsdata`) were used.

The test dataset contains 2,500 images that were not used during training or model selection.

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
- `classification_report`: provides classification metrics, including precision, recall, and F1-score.

### 8.3. General Metrics

The model was evaluated using the 2,500 images from the independent test dataset.

The following results were obtained:

| Metric       | Result |
| :----------- | -----: |
| Test Loss    | 0.2581 |
| Test Accuracy | 88.76% |
| Correct Predictions | 2,219 |
| Incorrect Predictions | 281 |

The model correctly classified 2,219 out of 2,500 images, achieving an accuracy of 88.76%.

### 8.4. Confusion Matrix

The confusion matrix was generated to examine the model's classification results for each category.

```text
[[1061  189]
 [  92 1158]]
```

The matrix is interpreted as follows:

| Actual Class | Predicted Cats | Predicted Dogs |
| :----------- | -------------: | -------------: |
| Cats         | 1,061          | 189            |
| Dogs         | 92             | 1,158          |

The model correctly classified:

- 1,061 cat images.
- 1,158 dog images.

It incorrectly classified:

- 189 cat images as dogs.
- 92 dog images as cats.

### 8.5. Classification Report

The classification report provided the following metrics:

```text
              precision    recall  f1-score   support

        cats       0.92      0.85      0.88      1250
        dogs       0.86      0.93      0.89      1250

    accuracy                           0.89      2500
   macro avg       0.89      0.89      0.89      2500
weighted avg       0.89      0.89      0.89      2500
```

The main results can be summarized as follows:

- **Cats:** 92% precision, 85% recall, and 88% F1-score.
- **Dogs:** 86% precision, 93% recall, and 89% F1-score.

The model achieved higher recall for dogs, meaning it correctly identified a greater proportion of actual dog images. For cats, it achieved higher precision, meaning that predictions classified as cats were more likely to be correct.

### 8.6. Result

The final evaluation confirmed that the model achieved an accuracy of 88.76% on the independent test dataset.

The results demonstrate that the EfficientNetB0-based classifier can distinguish between cats and dogs with good overall performance.

The difference between the validation accuracy (91.48%) and test accuracy (88.76%) is 2.72 percentage points. This provides an indication of the model's generalization performance on previously unseen images.

## 9. Individual Predictions

### 9.1. Objective

In addition to evaluating the model using the test dataset, a feature was implemented to allow users to provide an individual image and obtain a prediction.

This makes it possible to test the classifier using manually selected images and observe the predicted category and its associated confidence score.

### 9.2. Implementation

The following file was created:

```text id="f6y2pk"
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

```bash id="1nqj2f"
python src/predict.py
```

The program requests the image path:

```text id="h4m2s1"
Enter the image path:
```

The user enters the corresponding path, and the model returns the prediction.

### 9.4. Tests Performed

Two individual image predictions were performed to verify the functionality.

The first test used an image from the original training dataset, while the second used an image from the independent test dataset.

The following results were obtained:

| Image | Dataset | Predicted Category | Confidence |
| :--- | :--- | :--- | ---: |
| `cat.10.jpg` | Training | Cats | 99.22% |
| `cat.1.jpg` | Test | Cats | 99.95% |

**Example 1: Image from the training dataset**

```text
Enter the image path: data/raw/dogs-vs-cats/train/cat.10.jpg

Predicted class: cats
Confidence: 99.22%
```

**Example 2: Image from the test dataset**

```text
Enter the image path: data/processed_split/test/cats/cat.1.jpg

Predicted class: cats
Confidence: 99.95%
```

Both images were classified as cats, with high confidence scores.

### 9.5. Result

The individual prediction functionality was successfully implemented, allowing the trained model to classify images independently of the batch evaluation process.

The displayed confidence percentage represents the probability assigned by the model to its predicted category. It does not guarantee that the prediction is correct.

The second test also demonstrated that the model could make an individual prediction using an image from the reserved test dataset, which was not used during training or model selection.

## 10. Problems Encountered and Solutions

During project development, several challenges arose related to the execution environment, dependencies, and available resources.

### 10.1. RAM Limitations

**Problem:**

During the initial training tests, a batch size of 32 images was used. The process was terminated by the Codespaces environment, possibly related to resource limitations in the Codespaces environment. The exact cause was not confirmed.

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

Organizing the dataset required separating the original images into training, validation, and test directories, which could lead to file duplication and unnecessary storage consumption.

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
- Splitting the data into independent training, validation, and test sets.
- Exploring and visualizing the images.
- Implementing data generators using `ImageDataGenerator`.
- Building and compiling an EfficientNetB0-based neural network.
- Training the model using `ModelCheckpoint` and `EarlyStopping`.
- Evaluating the model using classification metrics and a confusion matrix.
- Implementing a script for individual image predictions.
- Documenting the development process and technical solutions.

### 11.2. Results Summary

The final model was evaluated using an independent test set containing 2,500 images.

| Indicator | Result |
| :--- | ---: |
| Original images | 25,000 |
| Training images | 20,000 |
| Validation images | 2,500 |
| Test images | 2,500 |
| Architecture | EfficientNetB0 |
| Total parameters | 4,213,797 |
| Best validation accuracy | 91.48% |
| Independent test accuracy | 88.76% |
| Test loss | 0.2581 |
| Correct predictions | 2,219 |
| Incorrect predictions | 281 |
| Saved model | `best_model.keras` |

The model correctly classified 2,219 out of 2,500 test images, achieving an accuracy of 88.76%.

The confusion matrix and classification report provide additional information about the model's performance on each category. The model achieved a recall of 85% for cats and 93% for dogs.

The best validation accuracy was 91.48%, achieved during epoch 8. The saved checkpoint was selected using `val_accuracy`, while `EarlyStopping` monitored `val_loss`.

### 11.3. Main Project Files

The project is organized into separate scripts, each responsible for a specific stage of the classification pipeline:

- `src/prepare_dataset.py`: prepares the dataset and creates the training, validation, and test splits.
- `src/data_loader.py`: loads the images and creates the data generators.
- `src/model.py`: defines and compiles the EfficientNetB0-based model.
- `src/train.py`: trains the model and applies the callbacks.
- `src/evaluate.py`: evaluates the saved model using the independent test set.
- `src/predict.py`: performs individual image predictions.
- `models/best_model.keras`: stores the best model checkpoint selected during training.

### 11.4. Limitations and Possible Improvements

Although the project was completed successfully, there are several opportunities for improvement:

1. **Pretrained weights:** experiment with EfficientNetB0 using ImageNet pretrained weights through transfer learning.
2. **Data augmentation:** incorporate data augmentation techniques to improve the model's generalization ability.
3. **Training optimization:** experiment with different batch sizes, optimizers, and learning rates.
4. **Results visualization:** incorporate training and validation accuracy/loss plots, as well as a visualization of the confusion matrix.
5. **Model performance:** investigate the classification errors, particularly the cat images incorrectly classified as dogs, to identify potential improvements.

### 11.5. Conclusion

This project made it possible to implement a complete image classification workflow using deep learning techniques.

The dataset was divided into independent training, validation, and test sets, ensuring that the final evaluation was performed on data that had not been used for training or checkpoint selection.

The EfficientNetB0-based classifier achieved an accuracy of 88.76% on the independent test set. The results demonstrate that the model can distinguish between cats and dogs with satisfactory performance, although some classification errors remain.

Individual prediction tests also confirmed that the prediction script works with specific image files.

The development process provided practical experience with Python, TensorFlow, Keras, image processing, model training and evaluation, and project management using Git and GitHub.
---