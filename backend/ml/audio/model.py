import torch
import torch.nn as nn
import librosa
import numpy as np
import io

# Audio CNN Model
class BirdAudioCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def audio_to_spectrogram(audio_bytes: bytes) -> np.ndarray:
    """Convert audio bytes to mel spectrogram."""
    audio_file = io.BytesIO(audio_bytes)
    y, sr = librosa.load(audio_file, sr=22050, duration=5.0, mono=True)

    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, fmax=8000
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize to 0-1
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-6)
    return log_mel


def load_audio_model(num_classes: int, model_path: str = None):
    """Load the audio CNN model."""
    model = BirdAudioCNN(num_classes=num_classes)

    if model_path:
        import os
        if os.path.exists(model_path):
            model.load_state_dict(
                torch.load(model_path, map_location="cpu")
            )
            print(f"Audio model loaded from {model_path}")
        else:
            print("No saved audio model found, using random weights")
    else:
        print("Audio model initialized with random weights")

    model.eval()
    return model


def predict_audio(audio_bytes: bytes, model, class_names: list):
    """Run inference on audio and return top 5 predictions."""
    spectrogram = audio_to_spectrogram(audio_bytes)

    # Resize to fixed shape (128 x 128)
    import torch.nn.functional as F
    tensor = torch.tensor(spectrogram, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(128, 128), mode="bilinear", align_corners=False)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)
        top5    = torch.topk(probs, min(5, len(class_names)))

    results = []
    for i, p in zip(top5.indices[0], top5.values[0]):
        results.append({
            "species"   : class_names[i.item()],
            "confidence": round(float(p) * 100, 2)
        })

    return results