"""Test script to debug model predictions"""
import torch
import sys
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b3
import os

BASE_DIR = Path(__file__).resolve().parent
CLASSES = ["Early-Blight", "Healthy-Leaf", "Late-Blight", "Leaf-Roll-Virus", "Virus-X"]

# Load model using the exact same method as main.py
print("Loading model...")
model_path = BASE_DIR / "best_efficientnet_b3.pth"
model = efficientnet_b3(weights=None, num_classes=len(CLASSES))

state_dict = torch.load(str(model_path), map_location='cpu')

# Remap classifier weights from the old format to torchvision format
new_state_dict = {}
for key, val in state_dict.items():
    if key.startswith('features.') or key.startswith('classifier.'):
        new_state_dict[key] = val

print(f"Loading {len(new_state_dict)} weights...")
missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
print(f"Missing: {missing}")
print(f"Unexpected: {unexpected}")
print("Model loaded!")

# Check if classifier was loaded
print(f"\nClassifier: {model.classifier}")
print(f"Classifier[1] weight: {model.classifier[1].weight[0, :5]}")
print(f"Classifier[1] bias: {model.classifier[1].bias}")
print(f"\nSaved classifier.1.weight[0, :5]: {state_dict['classifier.1.weight'][0, :5]}")

# Test transforms
transform_normalized = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model.eval()

# Test with a sample image from each class
data_dir = BASE_DIR / "Augmented_Dataset_Zip"

for class_name in CLASSES:
    class_dir = data_dir / class_name
    if not class_dir.exists():
        print(f"Skip {class_name}: directory not found")
        continue
    
    # Get first image
    images = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png'))]
    if not images:
        print(f"Skip {class_name}: no images")
        continue
    
    img_path = class_dir / images[0]
    image = Image.open(img_path).convert('RGB')
    
    # Test with normalization
    img_tensor = transform_normalized(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    pred_idx = probs.argmax().item()
    pred_class = CLASSES[pred_idx]
    confidence = probs[pred_idx].item() * 100
    
    print(f"\n{class_name} (actual)")
    print(f"  Predicted: {pred_class} ({confidence:.1f}%)")
    print(f"  All probs: {dict(zip(CLASSES, [f'{p:.1f}%' for p in (probs * 100).tolist()]))}")

print("\n\nDone!")

