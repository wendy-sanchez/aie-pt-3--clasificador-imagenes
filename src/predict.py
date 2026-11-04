
import numpy as np

from keras.models import load_model
from keras.utils import load_img, img_to_array

# Image dimensions
IMAGE_SIZE = (224, 224)

# Load the trained model
model = load_model("models/best_model.keras")

# Ask for the image path
image_path = input("Enter the image path: ")

# Load and resize the image
image = load_img(image_path, target_size=IMAGE_SIZE)

# Convert the image into a numerical array
image_array = img_to_array(image)

# Add a batch dimension
image_array = np.expand_dims(image_array, axis=0)

# Make a prediction
prediction = model.predict(image_array)

# Get the predicted class
predicted_class = np.argmax(prediction, axis=1)[0]

# Define class names
class_names = ["cats", "dogs"]

# Get prediction confidence
confidence = prediction[0][predicted_class] * 100

# Display the result
print(f"\nPredicted class: {class_names[predicted_class]}")
print(f"Confidence: {confidence:.2f}%")
