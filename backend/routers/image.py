import sys
import os
import json
from fastapi import APIRouter, UploadFile, File, HTTPException
from ml.image.model import load_model, predict_image

router = APIRouter()

CLASS_NAMES_PATH = os.path.join("ml", "image", "class_names.json")
MODEL_PATH       = os.path.join("ml", "image", "bird_model.pt")

_model       = None
_class_names = None

def get_model():
    global _model, _class_names
    if _model is None:
        with open(CLASS_NAMES_PATH, "r") as f:
            _class_names = json.load(f)
        print("Loading image model...")
        sys.stdout.flush()
        _model = load_model(num_classes=len(_class_names), model_path=MODEL_PATH)
        print("Image model ready!")
        sys.stdout.flush()
    return _model, _class_names

@router.post("/")
async def identify_bird_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG images are supported")

    image_bytes = await file.read()
    model, class_names = get_model()

    try:
        predictions = predict_image(image_bytes, model, class_names)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"filename": file.filename, "predictions": predictions}