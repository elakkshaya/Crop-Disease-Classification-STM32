import tensorflow as tf

# Dataset paths
TRAIN_PATH = r"E:\AI Projects\Crop_Disease_Project\dataset\PlantVillage\train"
VAL_PATH = r"E:\AI Projects\Crop_Disease_Project\dataset\PlantVillage\val"

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Load Training Dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Load Validation Dataset
val_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nTraining Dataset Loaded Successfully!")
print("Validation Dataset Loaded Successfully!")

print("\nClass Names:\n")
print(train_dataset.class_names)

print("\nNumber of Classes:", len(train_dataset.class_names))