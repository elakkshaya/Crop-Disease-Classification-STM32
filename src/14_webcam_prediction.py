import cv2
import numpy as np
import tensorflow as tf

# =========================================================
# PATHS
# =========================================================

MODEL_PATH = r"E:\AI Projects\Crop_Disease_Project\models\finetuned_model.keras"

# =========================================================
# LOAD MODEL
# =========================================================

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# =========================================================
# CLASS NAMES
# =========================================================

class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# =========================================================
# START WEBCAM
# =========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("\n========================================")
print("PLANT DISEASE WEBCAM DETECTION")
print("========================================")
print("Press Q to quit.")
print("Press SPACE to capture and classify.")
print("========================================\n")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame.")
        break

    # Display instructions
    cv2.putText(
        frame,
        "SPACE = Capture | Q = Quit",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("Plant Disease Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord("q"):
        break

    # Capture
    if key == 32:

        # Resize
        image = cv2.resize(frame, (224, 224))

        # Convert BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to float
        image = image.astype(np.float32)

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        # IMPORTANT:
        # Our model already contains Rescaling,
        # so DO NOT use preprocess_input() here.

        predictions = model.predict(image, verbose=0)

        class_id = np.argmax(predictions[0])
        confidence = predictions[0][class_id] * 100

        disease = class_names[class_id]

        print("\n================================")
        print("PREDICTION")
        print("================================")
        print("Disease/Class :", disease)
        print(f"Confidence    : {confidence:.2f}%")
        print("================================")

        # Show result
        result_frame = frame.copy()

        cv2.putText(
            result_frame,
            disease,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            result_frame,
            f"Confidence: {confidence:.2f}%",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("Prediction", result_frame)

        cv2.waitKey(2000)

cap.release()
cv2.destroyAllWindows()