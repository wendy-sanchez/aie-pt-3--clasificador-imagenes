import numpy as np

from keras.models import load_model
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from data_loader import tsdata

# Load the best saved model
model = load_model("models/best_model.keras")

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(tsdata)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Reset the test generator before making predictions
tsdata.reset()

# Generate predictions
predictions = model.predict(tsdata)

# Convert predictions into class labels
predicted_classes = np.argmax(predictions, axis=1)

# Get the true class labels
true_classes = tsdata.classes

# Get class names
class_names = list(tsdata.class_indices.keys())

# Calculate accuracy
accuracy = accuracy_score(true_classes, predicted_classes)

print(f"\nAccuracy Score: {accuracy * 100:.2f}%")

# Display the confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(true_classes, predicted_classes))

# Display the classification report
print("\nClassification Report:")
print(classification_report(
    true_classes,
    predicted_classes,
    target_names=class_names
))
