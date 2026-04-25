import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import os

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

def clean_species_name(raw_name: str) -> str:
    if "_" in raw_name:
        parts = raw_name.split("_", 1)
        return parts[1].strip().title()
    return raw_name.strip().title()

def load_model(num_classes: int, model_path: str = None):
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(model.fc.in_features, num_classes)
    )

    if model_path and os.path.exists(model_path):
        model.load_state_dict(
            torch.load(model_path, map_location="cpu")
        )
        print("Trained model loaded from " + model_path)
    else:
        print("No trained model found at " + str(model_path))

    model.eval()
    return model

def predict_image(image_bytes: bytes, model, class_names: list):
    image  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)
        top5    = torch.topk(probs, min(5, len(class_names)))

    results = []
    for i, p in zip(top5.indices[0], top5.values[0]):
        raw_name     = class_names[i.item()]
        clean_name   = clean_species_name(raw_name)
        results.append({
            "species":    clean_name,
            "confidence": round(float(p) * 100, 2)
        })

    return results