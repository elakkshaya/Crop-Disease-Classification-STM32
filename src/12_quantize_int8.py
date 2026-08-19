import tensorflow as tf
import os

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = r"E:\AI Projects\Crop_Disease_Project\models\finetuned_model.keras"

TRAIN_PATH = r"E:\AI Projects\Crop_Disease_Project\dataset\PlantVillage\train"

OUTPUT_MODEL = r"E:\AI Projects\Crop_Disease_Project\models\plant_disease_int8.tflite"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 1

# -----------------------------
# Load model
# -----------------------------
print("Loading fine-tuned Keras model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# -----------------------------
# Representative dataset
# -----------------------------
representative_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

def representative_dataset():

    for images, labels in representative_data.take(100):

        # IMPORTANT:
        # Do NOT use MobileNetV2 preprocess_input here.
        #
        # The model already contains a Rescaling layer.
        #
        images = tf.cast(images, tf.float32)

        yield [images.numpy()]

# -----------------------------
# Convert to INT8
# -----------------------------
print("\nStarting INT8 quantization...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# -----------------------------
# Convert
# -----------------------------
tflite_model = converter.convert()

# -----------------------------
# Save
# -----------------------------
with open(OUTPUT_MODEL, "wb") as f:
    f.write(tflite_model)

size_mb = os.path.getsize(OUTPUT_MODEL) / (1024 * 1024)

print("\n================================")
print("INT8 QUANTIZATION COMPLETE")
print("================================")

print("\nSaved to:")
print(OUTPUT_MODEL)

print(f"\nINT8 Model size: {size_mb:.2f} MB")