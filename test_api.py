"""Test API endpoint"""
import requests
import time
from pathlib import Path

time.sleep(2)  # Wait for server to start

try:
    # Test health
    resp = requests.get("http://127.0.0.1:8000/health", timeout=5)
    print(f"Health: {resp.status_code}")
    print(f"Response: {resp.json()}")
    
    # Test home
    resp = requests.get("http://127.0.0.1:8000/", timeout=5)
    print(f"\nHome page: {resp.status_code}")
    print(f"Content length: {len(resp.text)}")
    
    # Test prediction with a sample image
    import os
    data_dir = Path(__file__).parent / "Augmented_Dataset_Zip"
    if data_dir.exists():
        early_blight_dir = data_dir / "Early-Blight"
        if early_blight_dir.exists():
            images = [f for f in os.listdir(early_blight_dir) if f.endswith(('.jpg', '.png'))]
            if images:
                img_path = early_blight_dir / images[0]
                print(f"\nTesting with image: {img_path}")
                
                with open(img_path, 'rb') as f:
                    files = {'file': (img_path.name, f, 'image/jpeg')}
                    resp = requests.post("http://127.0.0.1:8000/predict", files=files, timeout=30)
                    print(f"Prediction: {resp.status_code}")
                    if resp.status_code == 200:
                        result = resp.json()
                        print(f"Predicted: {result['predicted_class']} ({result['confidence']}%)")
                        print(f"All predictions: {result['all_predictions']}")
                    else:
                        print(f"Error: {resp.text}")
            else:
                print(f"No images found in {early_blight_dir}")
        else:
            print(f"Directory not found: {early_blight_dir}")
    else:
        print(f"Data directory not found: {data_dir}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
