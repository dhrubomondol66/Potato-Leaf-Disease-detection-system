"""
Potato Leaf Disease Detection Web Application
FastAPI backend with EfficientNet-B3 model
"""
# uvicorn main:app --reload

import os
from io import BytesIO
from pathlib import Path

import torch
import timm
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b3
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request

# Initialize FastAPI app
app = FastAPI(
    title="Potato Leaf Disease Detection",
    description="Detect diseases in potato leaves using AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates and static files
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Mount static files if directory exists
if (BASE_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Disease classes
CLASSES = ["Early-Blight", "Healthy-Leaf", "Late-Blight", "Leaf-Roll-Virus", "Virus-X"]

# Disease information
DISEASE_INFO = {
    "Early-Blight": {
        "description": "A fungal disease caused by Alternaria solani",
        "symptoms": "Dark brown spots with concentric rings on leaves",
        "treatment": "Apply fungicides, practice crop rotation, remove infected plants"
    },
    "Healthy-Leaf": {
        "description": "No disease detected - leaf is healthy",
        "symptoms": "None - leaf appears normal",
        "treatment": "Continue regular plant care"
    },
    "Late-Blight": {
        "description": "A devastating disease caused by Phytophthora infestans",
        "symptoms": "Water-soaked lesions, white fungal growth on undersides",
        "treatment": "Apply fungicides immediately, destroy infected plants"
    },
    "Leaf-Roll-Virus": {
        "description": "A viral disease transmitted by aphids",
        "symptoms": "Upward rolling of leaves, stunted growth",
        "treatment": "Control aphid populations, remove infected plants"
    },
    "Virus-X": {
        "description": "Potato virus X (PVX) - a common viral infection",
        "symptoms": "Mild mosaic patterns, leaf curling",
        "treatment": "Use certified disease-free seed potatoes"
    }
}

# Model and transforms
model = None

# Normalization transform (for models trained with ImageNet normalization)
transform_normalized = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Simple transform without normalization (in case model was trained without it)
transform_simple = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def load_model():
    """Load the EfficientNet-B3 model using torchvision"""
    global model
    if model is None:
        print("Loading model...")
        model_path = BASE_DIR / "best_efficientnet_b3.pth"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load using torchvision's EfficientNet-B3
        model = efficientnet_b3(weights=None, num_classes=len(CLASSES))
        
        # Load the saved weights
        state_dict = torch.load(str(model_path), map_location='cpu')
        
        # Remap classifier weights from the old format to torchvision format
        # Old: classifier.1.weight, classifier.1.bias
        # New: classifier.1.weight, classifier.1.bias (same structure!)
        new_state_dict = {}
        for key, val in state_dict.items():
            if key.startswith('features.') or key.startswith('classifier.'):
                new_state_dict[key] = val
        
        # Load with strict=False to allow partial loading
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        print(f"Model loaded! Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
        model.eval()
    return model


def predict_disease(image: Image.Image) -> dict:
    """Predict disease from an image"""
    # Ensure RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Get model
    model = load_model()
    
    # Try with normalization first
    img_tensor = transform_normalized(image).unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    # If predictions are all very low confidence, try without normalization
    max_prob = probabilities.max().item()
    if max_prob < 0.3:  # If model is unsure, try without normalization
        print(f"Low confidence with normalization ({max_prob:.2%}), trying without...")
        img_tensor_simple = transform_simple(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor_simple)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    # Get predictions
    predicted_idx = probabilities.argmax().item()
    predicted_class = CLASSES[predicted_idx]
    confidence = probabilities[predicted_idx].item() * 100
    
    # All predictions with percentages
    all_predictions = {
        CLASSES[i]: round(probabilities[i].item() * 100, 2)
        for i in range(len(CLASSES))
    }
    
    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "all_predictions": all_predictions,
        "disease_info": DISEASE_INFO.get(predicted_class, {})
    }


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict disease from uploaded image"""
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        if not contents:
            raise ValueError("Empty file uploaded")
        
        image = Image.open(BytesIO(contents))
        
        # Get prediction
        result = predict_disease(image)
        
        return {
            "success": True,
            "filename": file.filename,
            **result
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}


if __name__ == "__main__":
    import uvicorn
    # Preload model on startup
    try:
        load_model()
        print("Model preloaded successfully!")
    except Exception as e:
        print(f"Warning: Could not preload model: {e}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
