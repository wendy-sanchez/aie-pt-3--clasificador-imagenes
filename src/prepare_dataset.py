import os
import random

# Reproducible random distribution
random.seed(42)

# Source and destination folders
source_dir = "data/raw/dogs-vs-cats/train"
output_dir = "data/processed_split"

# Training / validation / testing split
train_ratio = 0.80
validation_ratio = 0.10

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
    train_end = int(len(images) * train_ratio)
    validation_end = train_end + int(len(images) * validation_ratio)

    split_images = {
        "train": images[:train_end],
        "validation": images[train_end:validation_end],
        "test": images[validation_end:]
    }

    # Create hard links to avoid duplicating image files
    for split, split_category_images in split_images.items():
        destination_dir = os.path.join(output_dir, split, folder_name)
        os.makedirs(destination_dir, exist_ok=True)

        for image in split_category_images:
            source = os.path.join(source_dir, image)
            destination = os.path.join(destination_dir, image)
            if os.path.exists(destination):
                if not os.path.samefile(source, destination):
                    raise FileExistsError(
                        f"Existing destination does not match source: {destination}"
                    )
            else:
                os.link(source, destination)

    print(f"{folder_name}:")
    for split, split_category_images in split_images.items():
        print(f"  {split.capitalize()}: {len(split_category_images)}")
