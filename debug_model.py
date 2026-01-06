"""
Investigate model architecture and rebuild classifier
"""
import torch
import timm
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CLASSES = ["Early-Blight", "Healthy-Leaf", "Late-Blight", "Leaf-Roll-Virus", "Virus-X"]

# Load the saved state dict
model_path = BASE_DIR / "best_efficientnet_b3.pth"
state_dict = torch.load(str(model_path), map_location='cpu')

print("Saved model keys (first 30):")
saved_keys = list(state_dict.keys())[:30]
for i, key in enumerate(saved_keys):
    print(f"  {i}: {key}")

print(f"\nTotal keys in saved model: {len(state_dict.keys())}")

# Create new model
print("\n\nNew model structure:")
model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=len(CLASSES))
print(f"New model keys (first 30):")
model_keys = list(model.state_dict().keys())[:30]
for i, key in enumerate(model_keys):
    print(f"  {i}: {key}")

print(f"\nTotal keys in new model: {len(model.state_dict().keys())}")

# Check classifier keys
print("\n\nClassifier-related keys in saved model:")
classifier_keys = [k for k in state_dict.keys() if 'classifier' in k or 'fc' in k]
for key in classifier_keys:
    print(f"  {key}: shape {state_dict[key].shape}")

print("\n\nClassifier-related keys in new model:")
classifier_keys = [k for k in model.state_dict().keys() if 'classifier' in k or 'fc' in k]
for key in classifier_keys:
    print(f"  {key}: shape {model.state_dict()[key].shape}")
