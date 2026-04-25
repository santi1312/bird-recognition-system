import sys
import os
import tempfile
import subprocess
import torch
import torch.nn.functional as F
import json
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter()

_audio_model       = None
_audio_class_names = None

def get_audio_model():
    global _audio_model, _audio_class_names
    if _audio_model is None:
        from ml.audio.model import load_audio_model
        CLASS_NAMES_PATH = os.path.join("ml", "image", "class_names.json")
        with open(CLASS_NAMES_PATH, "r") as f:
            _audio_class_names = json.load(f)
        print("Loading audio model...")
        sys.stdout.flush()
        _audio_model = load_audio_model(num_classes=len(_audio_class_names))
        print("Audio model ready!")
        sys.stdout.flush()
    return _audio_model, _audio_class_names

ALLOWED_AUDIO_TYPES = [
    "audio/mpeg", "audio/wav", "audio/x-wav",
    "audio/mp3", "audio/ogg", "audio/flac",
    "application/octet-stream"
]

def convert_to_wav(input_path: str) -> str:
    output_path = input_path + "_converted.wav"
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", input_path,
         "-ar", "22050", "-ac", "1", "-c:a", "pcm_s16le", output_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise Exception("ffmpeg conversion failed: " + result.stderr)
    return output_path

@router.post("/")
async def identify_bird_audio(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    audio_bytes = await file.read()

    suffix = ".mp3"
    if file.filename.endswith(".wav"):    suffix = ".wav"
    elif file.filename.endswith(".ogg"):  suffix = ".ogg"
    elif file.filename.endswith(".flac"): suffix = ".flac"

    tmp_path = None
    wav_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        wav_path = convert_to_wav(tmp_path)

        import librosa
        y, sr = librosa.load(wav_path, sr=22050, duration=5.0, mono=True)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        log_mel  = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel  = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-6)

        tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        tensor = F.interpolate(tensor, size=(128, 128), mode="bilinear", align_corners=False)

        model, class_names = get_audio_model()

        with torch.no_grad():
            outputs = model(tensor)
            probs   = torch.softmax(outputs, dim=1)
            top5    = torch.topk(probs, min(5, len(class_names)))

        predictions = []
        for i, p in zip(top5.indices[0], top5.values[0]):
            raw_name = class_names[i.item()]
            name = raw_name.split("_", 1)[1].strip().title() if "_" in raw_name else raw_name.title()
            predictions.append({
                "species":    name,
                "confidence": round(float(p) * 100, 2)
            })

        return {"filename": file.filename, "predictions": predictions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path): os.unlink(tmp_path)
        if wav_path and os.path.exists(wav_path): os.unlink(wav_path)