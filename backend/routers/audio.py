import sys
import os
import tempfile
import subprocess
from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter()

_analyzer = None

def get_analyzer():
    global _analyzer
    if _analyzer is None:
        from birdnetlib import Recording
        from birdnetlib.analyzer import Analyzer
        print("Loading BirdNET analyzer...")
        sys.stdout.flush()
        _analyzer = Analyzer()
        print("BirdNET ready!")
        sys.stdout.flush()
    return _analyzer

ALLOWED_AUDIO_TYPES = [
    "audio/mpeg", "audio/wav", "audio/x-wav",
    "audio/mp3", "audio/ogg", "audio/flac",
    "application/octet-stream"
]

def convert_to_wav(input_path: str) -> str:
    output_path = input_path + "_converted.wav"
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", output_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise Exception("ffmpeg conversion failed: " + result.stderr)
    return output_path

@router.post("/")
async def identify_bird_audio(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported audio format: " + str(file.content_type))

    audio_bytes = await file.read()
    suffix = ".mp3"
    if file.filename.endswith(".wav"):   suffix = ".wav"
    elif file.filename.endswith(".ogg"): suffix = ".ogg"
    elif file.filename.endswith(".flac"): suffix = ".flac"

    tmp_path = None
    wav_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        wav_path = convert_to_wav(tmp_path)

        from birdnetlib import Recording
        analyzer = get_analyzer()

        recording = Recording(analyzer, wav_path, lat=0, lon=0, min_conf=0.1)
        recording.analyze()
        detections = recording.detections

        if not detections:
            return {"filename": file.filename, "predictions": [], "message": "No bird sounds detected"}

        predictions = []
        seen = set()
        for det in sorted(detections, key=lambda x: x["confidence"], reverse=True):
            species = det["common_name"]
            if species not in seen:
                seen.add(species)
                predictions.append({
                    "species":         species,
                    "scientific_name": det.get("scientific_name", ""),
                    "confidence":      round(det["confidence"] * 100, 2)
                })
            if len(predictions) >= 5:
                break

        return {"filename": file.filename, "predictions": predictions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path): os.unlink(tmp_path)
        if wav_path and os.path.exists(wav_path): os.unlink(wav_path)