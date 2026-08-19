import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

# -----------------------------
# Paths
# -----------------------------
VAL_PATH = r"E:\AI Projects\Crop_Disease_Project\dataset\PlantVillage\val"
MODEL_PATH = r"E:\AI Projects\Crop_Disease_Project\models\best_model.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# -----------------------------
# Load validation dataset
# -----------------------------
val_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = val_dataset.class_names

# -----------------------------
# Load model
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# Predictions
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
# Confusion matrix
# -----------------------------
cm = confusion_matrix(y_true, y_pred)

# -----------------------------
# Find misclassification pairs
# -----------------------------
errors = []

for true_class in range(len(class_names)):
    for predicted_class in range(len(class_names)):

        if true_class != predicted_class:

            count = cm[true_class][predicted_class]

            if count > 0:
                errors.append(
                    (
                        count,
                        class_names[true_class],
                        class_names[predicted_class]
                    )
                )

# Sort by number of mistakes
errors.sort(reverse=True)

# -----------------------------
# Display top 15
# -----------------------------
print("\n======================================")
print("TOP 15 MISCLASSIFICATION PAIRS")
print("======================================\n")

for i, (count, true_name, predicted_name) in enumerate(errors[:15], 1):

    print(
        f"{i:2}. {count:4} images | "
        f"Actual: {true_name} "
        f"--> Predicted: {predicted_name}"
    )

print("\nAnalysis Complete!")