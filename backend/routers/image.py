import sys
import os
import json
import torch
from fastapi import APIRouter, UploadFile, File, HTTPException
from ml.image.model import load_model, predict_image

router = APIRouter()

CLASS_NAMES_PATH = os.path.join("ml", "image", "class_names.json")
MODEL_PATH       = os.path.join("ml", "image", "bird_model.pt")

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

NUM_CLASSES = len(class_names)
print("Loading trained model with " + str(NUM_CLASSES) + " classes...")
sys.stdout.flush()

model = load_model(
    num_classes=NUM_CLASSES,
    model_path=MODEL_PATH
)

print("Image model ready!")
sys.stdout.flush()

@router.post("/")
async def identify_bird_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are supported"
        )

    image_bytes = await file.read()

    try:
        predictions = predict_image(image_bytes, model, class_names)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "filename":    file.filename,
        "predictions": predictions
    }