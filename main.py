import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms
import torchvision.datasets as datasets
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# 1. Custom Focal Loss Class
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 2. Model Definition
class PlantDiseaseClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        
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

# 3. Explainable AI (XAI) - Grad-CAM Function
def generate_xai_heatmap(model, input_tensor, original_image_np, target_category=None):
    target_layers = [model.base_model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_category)] if target_category is not None else None
    
    grayscale_cam = cam(input_tensor=input_tensor.unsqueeze(0), targets=targets)[0, :]
    visualization = show_cam_on_image(original_image_np, grayscale_cam, use_rgb=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_image_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(visualization)
    axes[1].set_title("Grad-CAM Explanation (XAI)")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

# 4. Main Execution Block
if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set path directly to the PlantVillage dataset directory
    DATA_DIR = r"C:\Users\admin\Downloads\archive\PlantVillage"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)

    # Dataset Diagnostics
    print(f"\n--- Dataset Info ---")
    print(f"Total Images: {len(full_dataset)}")
    print(f"Number of Classes: {len(full_dataset.classes)}")
    print(f"Class Names: {full_dataset.classes}")
    print(f"--------------------\n")

    if len(full_dataset.classes) <= 1:
        raise ValueError(
            "ImageFolder detected 1 or 0 classes! "
            "Please check that DATA_DIR points directly to the folder containing class directories."
        )

    # Split Dataset (80% Train, 20% Validation)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    num_classes = len(full_dataset.classes)
    model = PlantDiseaseClassifier(num_classes=num_classes).to(device)

    # Optimization Setup
    criterion = FocalLoss(alpha=1.0, gamma=2.0, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    epochs = 10

    # Training & Validation Loops
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Training")
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{epochs}] Validation")
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total

        scheduler.step(avg_val_loss)

        print(f"\nEpoch [{epoch+1}/{epochs}] Summary -> "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Accuracy: {val_accuracy:.2f}%\n")

    print("Training Complete! Demonstrating Grad-CAM (XAI) on a random validation image...")
    
    # Pick a random sample from the validation dataset each run
    random_idx = np.random.randint(0, len(val_dataset))
    sample_img_tensor, sample_label = val_dataset[random_idx]
    
    # Unnormalize image for visual display
    img_np = sample_img_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np, 0, 1)

    generate_xai_heatmap(
        model=model, 
        input_tensor=sample_img_tensor.to(device), 
        original_image_np=img_np, 
        target_category=sample_label
    )

    # Save model weights
    torch.save(model.state_dict(), "plant_disease_model.pth")
    print("Model saved successfully as plant_disease_model.pth!")