"""Check which weights actually got loaded"""
import torch
import torch.nn as nn
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "best_efficientnet_b3.pth"

state_dict = torch.load(str(model_path), map_location='cpu')

# Get classifier weights
print("Saved model classifier weights:")
print(f"  classifier.1.weight shape: {state_dict['classifier.1.weight'].shape}")
print(f"  classifier.1.weight sample: {state_dict['classifier.1.weight'][0, :5]}")
print(f"  classifier.1.bias: {state_dict['classifier.1.bias']}")

# Try to load with torchvision
from torchvision.models import efficientnet_b3
model = efficientnet_b3(pretrained=False, num_classes=5)

print(f"\nTorchvision model classifier:")
print(f"  model.classifier: {model.classifier}")
print(f"  classifier.weight shape: {model.classifier.weight.shape}")
print(f"  classifier weight sample: {model.classifier.weight[0, :5]}")

# Try loading
new_dict = {}
for key, val in state_dict.items():
    if key == 'classifier.1.weight':
        new_dict['classifier.weight'] = val
    elif key == 'classifier.1.bias':
        new_dict['classifier.bias'] = val
    elif key.startswith('features.') and 'classifier' not in key:
        new_dict[key] = val

print(f"\nLoading {len(new_dict)} weights...")
result = model.load_state_dict(new_dict, strict=False)
print(f"Missing keys: {len(result.missing_keys)}")
print(f"Unexpected keys: {len(result.unexpected_keys)}")

print(f"\nAfter loading, classifier.weight sample: {model.classifier.weight[0, :5]}")
print(f"After loading, classifier.bias: {model.classifier.bias}")

# Check if it actually changed
if torch.allclose(model.classifier.weight, state_dict['classifier.1.weight']):
    print("\n✓ Classifier weights loaded correctly!")
else:
    print("\n✗ Classifier weights were NOT loaded")
