# 🌱 Edge AI-Based Plant Disease Detection using MobileNetV2

An Edge AI-based plant disease classification system using **MobileNetV2**, transfer learning, fine-tuning, and **INT8 quantization** for lightweight deployment on resource-constrained embedded hardware.

---

## 📌 Project Overview

Plant diseases can significantly affect crop health and agricultural productivity. Early identification of disease symptoms can help farmers take appropriate action before the disease spreads.

This project develops a lightweight deep-learning-based plant disease classification system capable of classifying plant leaf images into **38 disease and healthy categories**.

The system uses **MobileNetV2 with transfer learning**, followed by fine-tuning to improve classification performance. The final model is converted to **TensorFlow Lite (TFLite)** and optimized using **full INT8 quantization** to reduce model size and computational requirements for future Edge AI deployment.

### Core Pipeline

```text
Leaf Image
    ↓
Image Preprocessing
    ↓
MobileNetV2
    ↓
Transfer Learning
    ↓
Fine-Tuning
    ↓
38-Class Disease Classification
    ↓
TensorFlow Lite Conversion
    ↓
Full INT8 Quantization
    ↓
ARM-Based Edge Deployment
🎯 Objectives
Develop an AI-based plant disease classification system.
Classify plant leaves into 38 disease/healthy categories.
Use MobileNetV2 for lightweight image classification.
Apply transfer learning to leverage pretrained visual features.
Fine-tune the final 30 MobileNetV2 layers for improved classification.
Evaluate the model using:
Accuracy
Precision
Recall
F1-score
Confusion Matrix
Convert the trained model to TensorFlow Lite.
Apply full INT8 quantization.
Reduce model size for resource-constrained deployment.
Prepare the optimized model for future ARM-based Edge AI hardware.
🧠 Technologies Used
Technology	Purpose
Python	Development
TensorFlow / Keras	Deep learning
MobileNetV2	CNN architecture
Transfer Learning	Initial model training
Fine-Tuning	Domain-specific improvement
TensorFlow Lite	Edge deployment format
INT8 Quantization	Model optimization
Scikit-learn	Evaluation metrics
Matplotlib	Visualization
VS Code	Development environment
Git / GitHub	Version control
📊 Dataset

The project uses the PlantVillage dataset.

Dataset Statistics
Total images: 54,303
Number of classes: 38
Training images: 43,444
Validation images: 10,861

The classes include healthy and diseased leaves from crops such as:

Apple
Blueberry
Cherry
Corn
Grape
Orange
Peach
Pepper
Potato
Raspberry
Soybean
Squash
Strawberry
Tomato

The dataset is not included in this repository.

🏗️ Model Architecture

The primary architecture used in this project is MobileNetV2.

MobileNetV2 is a lightweight Convolutional Neural Network designed for efficient image classification on devices with limited computational resources.

Architecture
Input Image
224 × 224 × 3
       ↓
Rescaling / Preprocessing
       ↓
MobileNetV2 Backbone
       ↓
Global Average Pooling
       ↓
Dropout
       ↓
Dense Layer
       ↓
38-Class Softmax Output
Model Configuration
Input size: 224 × 224 × 3
Backbone: MobileNetV2
Output classes: 38
Transfer learning: Yes
Fine-tuning: Last 30 MobileNetV2 layers
Optimized deployment format: TensorFlow Lite
Quantization: Full INT8
🔄 Transfer Learning

A pretrained MobileNetV2 network was used as the initial feature extractor.

Instead of training the complete CNN from scratch, the pretrained network provides general visual features such as edges, textures, shapes and patterns.

The classification head was adapted for the 38 PlantVillage classes.

This produced the initial baseline model.

Baseline Performance

Validation Accuracy: 93.37%

🔧 Fine-Tuning

After transfer learning, the final 30 layers of MobileNetV2 were made trainable.

Fine-tuning allows the pretrained network to adapt its learned features to the specific visual characteristics of plant diseases.

Improvement
Model	Validation Accuracy
Transfer Learning Baseline	93.37%
Fine-Tuned Model	96.05%

The fine-tuned model achieved:

Validation Accuracy: 96.05%
Macro Precision: 96.02%
Macro Recall: 94.49%
Macro F1-score: 95.03%
📈 Model Evaluation

The model was evaluated using:

Accuracy
Precision
Recall
F1-score
Confusion Matrix
Misclassification analysis

The confusion matrix was used to identify classes that were visually difficult to distinguish.

Examples of frequently observed misclassification pairs included:

Grape Esca
      ↕
Grape Black Rot

Corn Cercospora Leaf Spot
      ↕
Corn Northern Leaf Blight

Tomato Spider Mites
      ↕
Tomato Target Spot

Tomato Early Blight
      ↕
Tomato Septoria Leaf Spot

These results demonstrate that visually similar diseases remain challenging even after fine-tuning.

📦 Model Optimization

For embedded deployment, the fine-tuned model was converted to TensorFlow Lite.

Two deployment models were generated:

plant_disease_float32.tflite
plant_disease_int8.tflite
Model Sizes
Model	Size
Fine-tuned Keras model	~8.63 MB
Float32 TFLite	~9.05 MB
INT8 TFLite	2.63 MB
⚡ INT8 Quantization

Full INT8 quantization converts model numerical representations from higher precision floating-point values to 8-bit integer values.

This significantly reduces:

Model storage
Memory requirements
Computational requirements
Hardware resource requirements
Quantization Results
Metric	Fine-Tuned FP32	INT8
Validation Accuracy	96.05%	94.25%
Macro F1-score	95.03%	92.76%
Model Size	~8.63 MB	2.63 MB

The INT8 model therefore provides a practical trade-off between model size and classification performance.

🖥️ Current Deployment Status

The optimized model has been prepared for embedded deployment.

Current Status
✅ Dataset preparation
✅ Image preprocessing
✅ MobileNetV2 transfer learning
✅ Baseline training
✅ Baseline evaluation
✅ Fine-tuning
✅ Fine-tuned model evaluation
✅ TensorFlow Lite conversion
✅ INT8 quantization
✅ INT8 model evaluation
⬜ Camera integration
⬜ ARM hardware deployment
⬜ Real-world field validation
⬜ End-to-end latency measurement

The current project is therefore at the optimized-model / pre-hardware-deployment stage.

📷 Real-World Image Challenge

A major limitation of the PlantVillage dataset is that many images are captured under relatively controlled conditions.

Real-world agricultural images may contain:

Different lighting conditions
Complex backgrounds
Multiple leaves
Different camera angles
Shadows
Blur
Occlusion
Different distances from the camera

To improve robustness, the training pipeline incorporates image preprocessing and augmentation.

However, real-world field performance has not yet been quantitatively validated.

Future validation will involve images captured using a phone, laptop camera, or embedded camera under real environmental conditions.

🌐 Edge AI Approach

The long-term objective is to perform inference locally on ARM-based embedded hardware.

Instead of:

Camera → Internet → Cloud Server → Prediction

the proposed architecture is:

Camera
   ↓
Local Image Processing
   ↓
INT8 TFLite Model
   ↓
Local Prediction
   ↓
Disease Result

This can reduce dependence on:

Internet connectivity
Cloud infrastructure
High-performance computers

and supports the development of low-resource agricultural AI systems.

🔬 Literature / Research Gap Addressed

The project focuses on several practical gaps commonly encountered in plant disease detection systems:

1. Computational Requirements

Large CNN architectures can require significant computational resources.

Approach:
Use lightweight MobileNetV2 for efficient feature extraction.

2. Model Size

Large models can be difficult to store and execute on embedded devices.

Approach:
Apply TensorFlow Lite conversion and full INT8 quantization.

3. Limited Edge Deployment

High classification accuracy alone does not guarantee suitability for embedded deployment.

Approach:
Evaluate the model not only by classification performance but also by optimized model size.

4. Accuracy–Efficiency Trade-off

Quantization can reduce model size while potentially affecting accuracy.

Approach:
Compare the original fine-tuned model with the INT8 model.

5. Real-Time Local Inference

Cloud-based approaches may require continuous connectivity.

Approach:
Prepare a compact INT8 model for future local ARM-based inference.

📁 Project Structure
Crop_Disease_Project/
│
├── data/
│   └── plantvillage/
│
├── models/
│   ├── baseline/
│   ├── best_model.keras
│   ├── final_model.keras
│   ├── finetuned_model.keras
│   ├── plant_disease_float32.tflite
│   └── plant_disease_int8.tflite
│
├── results/
│   ├── confusion_matrix.png
│   └── ...
│
├── src/
│   ├── training scripts
│   ├── evaluation scripts
│   ├── fine-tuning scripts
│   └── quantization scripts
│
├── venv/
│
├── requirements.txt
│
└── README.md

The exact file structure may vary depending on the current development version.

🚀 Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/Crop_Disease_Project.git
cd Crop_Disease_Project

Create a virtual environment:

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
▶️ Running the Project

The general workflow is:

1. Prepare Dataset
        ↓
2. Preprocess Images
        ↓
3. Train MobileNetV2
        ↓
4. Evaluate Baseline
        ↓
5. Fine-Tune Model
        ↓
6. Evaluate Fine-Tuned Model
        ↓
7. Convert to TensorFlow Lite
        ↓
8. Apply INT8 Quantization
        ↓
9. Evaluate INT8 Model
        ↓
10. Deploy on Edge Hardware
🧪 Evaluation Results
Baseline
Validation Accuracy: 93.37%
Fine-Tuned Model
Validation Accuracy: 96.05%
Macro F1-score:      95.03%
INT8 Model
Model Size:          2.63 MB
Validation Accuracy: 94.25%
Macro F1-score:      92.76%
Final Comparison
                    Accuracy       Model Size

Baseline            93.37%         —
Fine-Tuned          96.05%         ~8.63 MB
INT8                 94.25%         2.63 MB
🛠️ Future Work

The next stage of the project focuses on hardware and real-world validation.

Planned Work
 Integrate camera input.
 Deploy the INT8 TFLite model on ARM-based hardware.
 Test inference latency.
 Measure RAM and memory usage.
 Test power consumption.
 Capture real-world leaf images.
 Evaluate performance under different lighting/background conditions.
 Perform real-world validation.
 Investigate further optimization if required.
 Develop a farmer-facing prediction interface.
 Conduct end-to-end field testing.
👨‍🌾 Intended Farmer Workflow

The intended final system is designed to be simple for a farmer:

Step 1
Capture a leaf image
        ↓
Step 2
System preprocesses the image
        ↓
Step 3
INT8 MobileNetV2 performs local inference
        ↓
Step 4
System displays predicted disease
        ↓
Step 5
Farmer can use the result as an early identification aid

The system is intended as an AI-based classification and early identification tool, not as a replacement for professional agricultural diagnosis.

📌 Important Limitations
Current quantitative results are based on the PlantVillage validation dataset.
Real-world field-image accuracy has not yet been established.
Hardware deployment has not yet been completed.
End-to-end inference latency on the target ARM hardware has not yet been measured.
Different lighting, backgrounds, camera quality and leaf orientations may affect real-world performance.
The current model performs classification among the predefined 38 classes and cannot identify diseases outside those classes.
🏆 Key Achievement

The project demonstrates that a pretrained and fine-tuned MobileNetV2 model can achieve high classification performance while being compressed into a compact 2.63 MB INT8 TensorFlow Lite model.

54,303 Images
      ↓
38 Classes
      ↓
MobileNetV2
      ↓
Transfer Learning
      ↓
Fine-Tuning
      ↓
96.05% Validation Accuracy
      ↓
INT8 Quantization
      ↓
2.63 MB Model
      ↓
94.25% Validation Accuracy
      ↓
Future ARM Edge Deployment
📜 Conclusion

This project presents a lightweight Edge AI approach for automated plant disease classification.

MobileNetV2 transfer learning provided a strong baseline of 93.37% validation accuracy, while fine-tuning the final 30 layers improved performance to 96.05%.

For resource-constrained deployment, the model was converted to TensorFlow Lite and fully quantized to INT8, reducing the model size to 2.63 MB while maintaining 94.25% validation accuracy.

The results demonstrate a practical balance between classification performance and model efficiency, providing a foundation for future deployment on ARM-based embedded hardware and real-world agricultural applications.

📄 Project Status

Current Stage: Model development and optimization completed; hardware deployment and real-world validation pending.

Model: MobileNetV2
Classes: 38
Fine-Tuned Accuracy: 96.05%
INT8 Accuracy: 94.25%
INT8 Model Size: 2.63 MB
Target: ARM-based Edge AI deployment
