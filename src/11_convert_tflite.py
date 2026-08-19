import tensorflow as tf
import os

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = r"E:\AI Projects\Crop_Disease_Project\models\finetuned_model.keras"

OUTPUT_PATH = r"E:\AI Projects\Crop_Disease_Project\models\plant_disease_float32.tflite"

# -----------------------------
# Load model
# -----------------------------
print("Loading fine-tuned model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# -----------------------------
# Convert to TensorFlow Lite
# -----------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

# -----------------------------
# Save model
# -----------------------------
with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

# -----------------------------
# Model size
# -----------------------------
size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)

print("\n================================")
print("TFLITE CONVERSION COMPLETE")
print("================================")

print("\nSaved to:")
print(OUTPUT_PATH)

print(f"\nModel size: {size_mb:.2f} MB")