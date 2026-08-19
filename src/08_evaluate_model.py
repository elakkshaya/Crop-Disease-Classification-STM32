import tensorflow as tf
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------
# Paths
# -----------------------------
VAL_PATH = r"E:\AI Projects\Crop_Disease_Project\dataset\PlantVillage\val"
MODEL_PATH = r"E:\AI Projects\Crop_Disease_Project\models\finetuned_model.keras"

# -----------------------------
# Settings
# -----------------------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# -----------------------------
# Load Validation Dataset
# -----------------------------
val_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = val_dataset.class_names

print("\nNumber of Classes:", len(class_names))

# -----------------------------
# Load Trained Model
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)

print("\nModel Loaded Successfully!")

# -----------------------------
# Generate Predictions
# -----------------------------
y_true = []
y_pred = []

for images, labels in val_dataset:

    predictions = model.predict(images, verbose=0)

    predicted_classes = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# -----------------------------
# Accuracy
# -----------------------------
accuracy = accuracy_score(y_true, y_pred)

print("\n================================")
print("MODEL EVALUATION")
print("================================")

print(f"\nValidation Accuracy: {accuracy * 100:.2f}%")

# -----------------------------
# Classification Report
# -----------------------------
print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix Generated.")

# -----------------------------
# Save Confusion Matrix
# -----------------------------
os.makedirs(
    r"E:\AI Projects\Crop_Disease_Project\results",
    exist_ok=True
)

plt.figure(figsize=(18, 16))

sns.heatmap(
    cm,
    annot=False,
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("PlantVillage Disease Classification - Confusion Matrix")

plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig(
    r"E:\AI Projects\Crop_Disease_Project\results\confusion_matrix_finetuned.png",
    dpi=300
)

plt.close()

print(
    "\nConfusion matrix saved to:"
    "\nresults/confusion_matrix.png"
)

print("\nEvaluation Complete!")