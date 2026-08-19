import tensorflow as tf

# -----------------------------
# Paths
# -----------------------------
TRAIN_PATH = r"E:\AI Projects\Crop_Disease_Project\dataset\PlantVillage\train"
VAL_PATH = r"E:\AI Projects\Crop_Disease_Project\dataset\PlantVillage\val"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# -----------------------------
# Load Dataset
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

CLASS_NAMES = train_dataset.class_names

print("\nClasses:", len(CLASS_NAMES))

# -----------------------------
# Improve Performance
# -----------------------------
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
val_dataset = val_dataset.prefetch(AUTOTUNE)

# -----------------------------
# Data Augmentation
# -----------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.10),
    tf.keras.layers.RandomZoom(0.10),
])

# -----------------------------
# MobileNetV2
# -----------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# -----------------------------
# Build Model
# -----------------------------
model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Rescaling(1./255),

    base_model,

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(
        len(CLASS_NAMES),
        activation="softmax"
    )
])

# -----------------------------
# Compile
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Callbacks
# -----------------------------
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    r"E:\AI Projects\Crop_Disease_Project\models\best_model.keras",
    save_best_only=True,
    monitor="val_accuracy"
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True
)

# -----------------------------
# Train
# -----------------------------
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=2,
    callbacks=[checkpoint, early_stop]
)

# -----------------------------
# Save Final Model
# -----------------------------
model.save(
    r"E:\AI Projects\Crop_Disease_Project\models\final_model.keras"
)

print("\nTraining Finished Successfully!")