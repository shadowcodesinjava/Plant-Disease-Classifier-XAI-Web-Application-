# 🌱 Plant Disease Classifier & XAI (Explainable AI)

A deep learning pipeline and interactive web application for diagnosing plant diseases from leaf imagery. Powered by an EfficientNet-B0 backbone fine-tuned with Focal Loss, this project integrates Grad-CAM (Gradient-weighted Class Activation Mapping) to provide visual heatmaps that explain why the model makes specific predictions.

## 📌 Project Overview

While modern deep learning models can achieve near-perfect validation scores on benchmark agricultural datasets, they frequently act as "black boxes." This project addresses model opacity and class imbalance in agricultural computer vision by:

1. **Classifying 15 Leaf Categories**: Identifying healthy leaves and distinct fungal/bacterial diseases across Pepper, Potato, and Tomato plants.
2. **Mitigating Class Imbalance**: Utilizing Focal Loss instead of standard Cross-Entropy Loss to focus training on hard-to-classify and underrepresented disease classes.
3. **Visual Explainability (XAI)**: Generating real-time Grad-CAM spatial attention maps to highlight lesion features driving model predictions.
4. **Analyzing Domain Shift**: Evaluating performance gaps between studio-captured benchmark data (PlantVillage) and real-world field conditions (Google/In-the-Wild images).

## 🛠️ Architecture & Tech Stack

- **Framework**: PyTorch & Torchvision
- **Model Backbone**: EfficientNet-B0 (Pretrained Transfer Learning)
- **Custom Head**: `Dropout(0.3) -> Linear(in_features, 256) -> ReLU() -> Dropout(0.3) -> Linear(256, 15)`
- **Loss Function**: Focal Loss (γ = 2.0)
- **Explainability**: `pytorch-grad-cam` (Grad-CAM targeting final feature layer)
- **Web App**: Streamlit, OpenCV, PIL
- **Evaluation Metrics**: Validation Accuracy (99.66%), Softmax Confidence Scoring

## 🚀 Key Features

- **Real-Time Web Interface**: Simple drag-and-drop UI built with Streamlit for uploading leaf images.
- **Confidence Scoring**: Softmax output mapping providing mathematical certainty for predicted classes.
- **Grad-CAM Heatmap Visualization**: Dynamic side-by-side display of the original image alongside the spatial activation map.
- **Out-of-Distribution Warning Handling**: Softmax confidence monitoring to flag ambiguous or low-confidence inputs.

## 📂 Dataset Structure

The model is trained on 15 classes from the PlantVillage dataset:

- **Pepper (Bell)**: Bacterial Spot, Healthy
- **Potato**: Early Blight, Late Blight, Healthy
- **Tomato**: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites (Two-spotted), Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy

## 📦 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/plant-disease-classifier-xai.git
cd plant-disease-classifier-xai
```

### 2. Create a Virtual Environment & Install Dependencies

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install torch torchvision streamlit pytorch-grad-cam opencv-python pillow numpy
```

### 3. Run the Streamlit Application

Ensure your trained weights (`plant_disease_model.pth`) are placed in the root directory:

```bash
streamlit run app.py
```
