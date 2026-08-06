import os

# Dataset paths
TRAIN_PATH = r"E:\AI Projects\Crop_Disease_Project\dataset\PlantVillage\train"
VAL_PATH = r"E:\AI Projects\Crop_Disease_Project\dataset\PlantVillage\val"

# Count classes
train_classes = sorted(os.listdir(TRAIN_PATH))
val_classes = sorted(os.listdir(VAL_PATH))

print(f"Training classes   : {len(train_classes)}")
print(f"Validation classes : {len(val_classes)}")

print("\nFirst 10 classes:")
for cls in train_classes[:10]:
    print(cls)