import os
import numpy as np
from PIL import Image

# Dataset path
TRAIN_PATH = r"E:\AI Projects\Crop_Disease_Project\dataset\PlantVillage\train"

# Select first class
class_name = sorted(os.listdir(TRAIN_PATH))[0]
class_path = os.path.join(TRAIN_PATH, class_name)

# Select first image
image_name = sorted(os.listdir(class_path))[0]
image_path = os.path.join(class_path, image_name)

# Load image
img = Image.open(image_path)

print("Original Size :", img.size)

# Resize image
img = img.resize((224, 224))

print("Resized Size :", img.size)

# Convert to NumPy array
img_array = np.array(img)

print("\nShape :", img_array.shape)
print("Data Type :", img_array.dtype)

# Normalize
img_array = img_array / 255.0

print("\nMinimum Pixel Value :", img_array.min())
print("Maximum Pixel Value :", img_array.max())