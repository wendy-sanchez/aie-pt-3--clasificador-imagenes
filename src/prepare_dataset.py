import os
import random

# Reproducible random distribution
random.seed(42)

# Source and destination folders
source_dir = "data/raw/dogs-vs-cats/train"
output_dir = "data/processed"

# Training / testing split
train_ratio = 0.80

# Image categories
categories = {
    "cat": "cats",
    "dog": "dogs"
}

for category, folder_name in categories.items():

    # Get all images for this category
    images = [
        image for image in os.listdir(source_dir)
        if image.startswith(category + ".") and image.endswith(".jpg")
    ]

    # Shuffle images randomly
    random.shuffle(images)

    # Calculate split
    split_index = int(len(images) * train_ratio)

    train_images = images[:split_index]
    test_images = images[split_index:]

    # Create hard links to avoid duplicating image files
    for image in train_images:
        source = os.path.join(source_dir, image)
        destination = os.path.join(
            output_dir, "train", folder_name, image
        )
        os.link(source, destination)

    for image in test_images:
        source = os.path.join(source_dir, image)
        destination = os.path.join(
            output_dir, "test", folder_name, image
        )
        os.link(source, destination)

    print(f"{folder_name}:")
    print(f"  Training: {len(train_images)}")
    print(f"  Testing: {len(test_images)}")
