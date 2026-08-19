import tensorflow as tf

# Image settings
IMAGE_SIZE = (224, 224)

# Load MobileNetV2 without the final classifier
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

# Build our model
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),

    base_model,

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(
        38,
        activation="softmax"
    )
])

# Display summary
model.summary()