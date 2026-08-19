import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report

MODEL_PATH = r"E:\AI Projects\Crop_Disease_Project\models\plant_disease_int8.tflite"
VAL_PATH = r"E:\AI Projects\Crop_Disease_Project\dataset\PlantVillage\val"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 1

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

print("Number of Classes:", len(class_names))

# -----------------------------
# Load INT8 model
# -----------------------------
print("\nLoading INT8 TFLite model...")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("INT8 model loaded successfully.")

input_scale, input_zero_point = input_details[0]["quantization"]
output_scale, output_zero_point = output_details[0]["quantization"]

print("\nInput scale:", input_scale)
print("Input zero point:", input_zero_point)

print("Output scale:", output_scale)
print("Output zero point:", output_zero_point)

# -----------------------------
# Evaluation
# -----------------------------
y_true = []
y_pred = []

print("\n================================")
print("INT8 MODEL EVALUATION")
print("================================")

for images, labels in val_dataset:

    # IMPORTANT:
    # Do NOT use MobileNetV2 preprocess_input().
    # The Keras model contains its own Rescaling layer.

    image = images[0].numpy().astype(np.float32)

    # Quantize raw 0-255 image to INT8 input
    image = image / input_scale + input_zero_point

    image = np.round(image)

    image = np.clip(image, -128, 127)

    image = image.astype(np.int8)

    image = np.expand_dims(image, axis=0)

    # Inference
    interpreter.set_tensor(
        input_details[0]["index"],
        image
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details[0]["index"]
    )

    # Dequantize output
    output = (
        output.astype(np.float32) - output_zero_point
    ) * output_scale

    prediction = np.argmax(output[0])

    y_true.append(int(labels[0]))
    y_pred.append(int(prediction))

# -----------------------------
# Results
# -----------------------------
accuracy = np.mean(
    np.array(y_true) == np.array(y_pred)
)

print("\n================================")
print("RESULTS")
print("================================")

print(f"\nINT8 Validation Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)

print("\nINT8 Evaluation Complete!")