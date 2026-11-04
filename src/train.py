
import os

from keras.callbacks import ModelCheckpoint, EarlyStopping

from data_loader import trdata, tsdata
from model import model

# Create directory to save the best model
os.makedirs("models", exist_ok=True)

# Save the best model during training
checkpoint = ModelCheckpoint(
    filepath="models/best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

# Stop training when validation loss stops improving
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

# Train the model
history = model.fit(
    trdata,
    validation_data=tsdata,
    epochs=10,
    callbacks=[checkpoint, early_stopping]
)
