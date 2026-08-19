import tensorflow as tf

# -----------------------------
# Paths
# -----------------------------
TRAIN_PATH = r"E:\AI Projects\Crop_Disease_Project\dataset\PlantVillage\train"
VAL_PATH = r"E:\AI Projects\Crop_Disease_Project\dataset\PlantVillage\val"

BASE_MODEL_PATH = r"E:\AI Projects\Crop_Disease_Project\models\baseline\baseline_93_37.keras"
OUTPUT_PATH = r"E:\AI Projects\Crop_Disease_Project\models\finetuned_model.keras"

# -----------------------------
# Settings
# -----------------------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# -----------------------------
# Load datasets
# -----------------------------
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# -----------------------------
# Load baseline model
# -----------------------------
model = tf.keras.models.load_model(BASE_MODEL_PATH)

print("\nBaseline model loaded successfully!")

# -----------------------------
# Find MobileNetV2 by name
# -----------------------------
base_model = None

for layer in model.layers:
    print("Layer:", layer.name, "| Type:", type(layer).__name__)

    if "mobilenetv2" in layer.name.lower():
        base_model = layer
        break

if base_model is None:
    raise ValueError("MobileNetV2 backbone not found.")

print("\nMobileNetV2 backbone found:")
print(base_model.name)

# -----------------------------
# Unfreeze backbone
# -----------------------------
base_model.trainable = True

# Freeze most layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Fine-tune last 30 layers
for layer in base_model.layers[-30:]:
    layer.trainable = True

print("\nFine-tuning enabled.")
print("Last 30 MobileNetV2 layers are trainable.")

# -----------------------------
# Recompile
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# Fine-tune
# -----------------------------
print("\nStarting fine-tuning...\n")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5
)

# -----------------------------
# Save model
# -----------------------------
model.save(OUTPUT_PATH)

print("\n================================")
print("FINE-TUNING COMPLETE")
print("================================")

print("\nFine-tuned model saved to:")
print(OUTPUT_PATH)

print("\nFinal Training Accuracy:",
      history.history["accuracy"][-1])

print("Final Validation Accuracy:",
      history.history["val_accuracy"][-1])

print("\nOriginal 93.37% baseline remains safe.")