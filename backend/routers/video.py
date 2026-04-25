from fastapi import APIRouter, UploadFile, File, HTTPException
from ml.video.processor import extract_frames, frame_to_bytes, aggregate_predictions
from ml.image.model import load_model, predict_image
import json
import os

router = APIRouter()

CLASS_NAMES_PATH = os.path.join("ml", "image", "class_names.json")
MODEL_PATH       = os.path.join("ml", "image", "bird_model.pt")

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

video_model = load_model(
    num_classes=len(class_names),
    model_path=MODEL_PATH
)

ALLOWED_VIDEO_TYPES = [
    "video/mp4",
    "video/avi",
    "video/quicktime",
    "video/x-msvideo",
    "video/x-matroska",
    "application/octet-stream"
]

@router.post("/")
async def identify_bird_video(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format: {file.content_type}"
        )

    video_bytes = await file.read()

    try:
        # Step 1: Extract frames
        frames = extract_frames(video_bytes, num_frames=8)

        if not frames:
            raise HTTPException(
                status_code=400,
                detail="Could not extract frames from video"
            )

        # Step 2: Run image model on each frame
        all_predictions = []
        for frame in frames:
            frame_bytes = frame_to_bytes(frame)
            preds = predict_image(frame_bytes, video_model, class_names)
            all_predictions.append(preds)

        # Step 3: Aggregate predictions across frames
        final_predictions = aggregate_predictions(all_predictions)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "filename"      : file.filename,
        "frames_analyzed": len(frames),
        "predictions"   : final_predictions
    }