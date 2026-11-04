
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image dimensions
IMAGE_SIZE = (224, 224)

# Batch size
BATCH_SIZE = 8

# Dataset directories
train_dir = "data/processed/train"
test_dir = "data/processed/test"

# Create image data generators
train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

# Load training images
trdata = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

# Load testing images
tsdata = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)