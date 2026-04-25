import cv2
import numpy as np
import tempfile
import os
from collections import Counter

def extract_frames(video_bytes: bytes, num_frames: int = 8) -> list:
    """Extract evenly spaced frames from video bytes."""

    # Write video bytes to a temp file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    frames = []
    try:
        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            raise ValueError("Could not read video or video has no frames")

        interval = max(1, total_frames // num_frames)

        for i in range(0, total_frames, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            if len(frames) >= num_frames:
                break

        cap.release()
    finally:
        os.unlink(tmp_path)

    return frames


def frame_to_bytes(frame: np.ndarray) -> bytes:
    """Convert numpy frame array to JPEG bytes."""
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", frame_bgr)
    return buffer.tobytes()


def aggregate_predictions(all_predictions: list) -> dict:
    """
    Aggregate predictions from multiple frames.
    Uses weighted voting — higher confidence = more weight.
    """
    species_scores = {}

    for frame_preds in all_predictions:
        for pred in frame_preds:
            species = pred["species"]
            confidence = pred["confidence"]
            if species not in species_scores:
                species_scores[species] = 0.0
            species_scores[species] += confidence

    # Sort by total score
    sorted_species = sorted(
        species_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Normalize scores
    total = sum(score for _, score in sorted_species)
    results = []
    for species, score in sorted_species[:5]:
        results.append({
            "species"   : species,
            "confidence": round((score / total) * 100, 2)
        })

    return results