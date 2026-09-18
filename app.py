import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image

# 1. Page Configuration
st.set_page_config(
    page_title="Plant Disease Classifier", 
    page_icon="🌱", 
    layout="centered"
)

st.title("🌱 AI Plant Disease Classifier & XAI")
st.write("Upload a leaf image to run real-time disease identification powered by EfficientNet-B0.")

# 2. Model Architecture
class PlantDiseaseClassifier(nn.Module):
    def __init__(self, num_classes=15):
        super().__init__()
        self.base_model = efficientnet_b0(weights=None)
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

# 3. Class Names Mapping (15 Classes)
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot', 
    'Pepper__bell___healthy', 
    'Potato___Early_blight', 
    'Potato___Late_blight', 
    'Potato___healthy',
    'Tomato_Bacterial_spot', 
    'Tomato_Early_blight', 
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 
    'Tomato_Septoria_leaf_spot', 
    'Tomato_Spider_mites_Two_spotted_spider_mite', 
    'Tomato_Target_Spot',
    'Tomato_Tomato_YellowLeaf_Curl_Virus', 
    'Tomato_Tomato_mosaic_virus', 
    'Tomato_healthy'
]

# 4. Load Saved Weights
@st.cache_resource
def load_model():
    model = PlantDiseaseClassifier(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load("plant_disease_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# 5. Preprocessing Pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 6. User Interface & Prediction
uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Leaf Image", use_container_width=True)
    
    if st.button("Analyze Leaf", type="primary"):
        img_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_class = torch.max(probabilities, 0)

        result_label = CLASS_NAMES[predicted_class.item()].replace('_', ' ')
        confidence_pct = confidence.item() * 100

        st.markdown("---")
        st.success(f"**Prediction:** {result_label}")
        st.info(f"**Confidence Score:** {confidence_pct:.2f}%")