
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image dimensions
IMAGE_SIZE = (224, 224)

# Batch size
BATCH_SIZE = 8

# Dataset directories
dataset_dir = "data/processed_split"
train_dir = os.path.join(dataset_dir, "train")
validation_dir = os.path.join(dataset_dir, "validation")
test_dir = os.path.join(dataset_dir, "test")

# Create image data generators
train_datagen = ImageDataGenerator()
validation_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

# Load training images
trdata = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

# Load validation images
valdata = validation_datagen.flow_from_directory(
    directory=validation_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# Load testing images
tsdata = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)