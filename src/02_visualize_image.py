import os
from PIL import Image
import matplotlib.pyplot as plt

# Dataset path
TRAIN_PATH = r"E:\AI Projects\Crop_Disease_Project\dataset\PlantVillage\train"

# Select the first class
class_name = sorted(os.listdir(TRAIN_PATH))[0]
class_path = os.path.join(TRAIN_PATH, class_name)

# Select the first image
image_name = sorted(os.listdir(class_path))[0]
image_path = os.path.join(class_path, image_name)

# Open image
img = Image.open(image_path)

# Display information
print("Class       :", class_name)
print("Image Name  :", image_name)
print("Image Size  :", img.size)
print("Image Mode  :", img.mode)

# Display image
plt.imshow(img)
plt.title(class_name)
plt.axis("off")
plt.savefig("docs/sample_leaf.png", dpi=300)
print("Image saved to docs/sample_leaf.png")