import sys
import os

# Download model FIRST before anything else
print("Checking model...")
sys.stdout.flush()

from download_model import download_model
download_model()

print("Model check complete!")
sys.stdout.flush()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import engine, Base
import models.bird
import models.media
from routers import birds, image, audio, video

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Bird Recognition API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(birds.router, prefix="/api/birds",          tags=["Birds"])
app.include_router(image.router, prefix="/api/identify/image", tags=["Image Recognition"])
app.include_router(audio.router, prefix="/api/identify/audio", tags=["Audio Recognition"])
app.include_router(video.router, prefix="/api/identify/video", tags=["Video Recognition"])

@app.get("/")
def root():
    return {"message": "Bird Recognition API is running!"}