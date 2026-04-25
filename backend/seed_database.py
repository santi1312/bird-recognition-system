import sys
import os
import json
import time
import requests
from sqlalchemy.orm import Session
from database import engine, Base, SessionLocal
from models.bird import Bird
from models.media import Media

# Load class names from training
NAMES_PATH = os.path.join("ml", "image", "class_names.json")
with open(NAMES_PATH, "r") as f:
    class_names = json.load(f)

print("Loaded " + str(len(class_names)) + " bird species")
sys.stdout.flush()

def get_wikipedia_info(bird_name):
    try:
        search_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + bird_name.replace(" ", "_")
        response = requests.get(search_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                "description": data.get("extract", "")[:500],
                "image_url": data.get("thumbnail", {}).get("source", None)
            }
    except Exception as e:
        print("Wikipedia error for " + bird_name + ": " + str(e))
    return {"description": "", "image_url": None}

def get_xeno_canto_audio(bird_name):
    try:
        url = "https://xeno-canto.org/api/2/recordings?query=" + bird_name.replace(" ", "+")
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            recordings = data.get("recordings", [])
            if recordings:
                rec = recordings[0]
                audio_url = "https:" + rec.get("file", "")
                return audio_url
    except Exception as e:
        print("Xeno-canto error for " + bird_name + ": " + str(e))
    return None

def seed_birds():
    db = SessionLocal()
    try:
        existing = db.query(Bird).count()
        if existing > 0:
            print("Database already has " + str(existing) + " birds. Skipping.")
            return

        print("Starting database seeding...")
        sys.stdout.flush()

        for i, name in enumerate(class_names):
            try:
                print(str(i+1) + "/" + str(len(class_names)) + " - Seeding: " + name)
                sys.stdout.flush()

                # Get Wikipedia info
                wiki = get_wikipedia_info(name)
                time.sleep(0.5)

                # Get xeno-canto audio
                audio_url = get_xeno_canto_audio(name)
                time.sleep(0.5)

                # Create bird record
                bird = Bird(
                    common_name=name.title(),
                    scientific_name="",
                    family="",
                    habitat="",
                    description=wiki["description"]
                )
                db.add(bird)
                db.flush()

                # Add image media
                if wiki["image_url"]:
                    image_media = Media(
                        bird_id=bird.id,
                        type="image",
                        url=wiki["image_url"],
                        source="Wikipedia"
                    )
                    db.add(image_media)

                # Add audio media
                if audio_url:
                    audio_media = Media(
                        bird_id=bird.id,
                        type="audio",
                        url=audio_url,
                        source="xeno-canto"
                    )
                    db.add(audio_media)

                db.commit()

            except Exception as e:
                print("Error seeding " + name + ": " + str(e))
                db.rollback()
                continue

        print("Seeding complete!")
        final_count = db.query(Bird).count()
        print("Total birds in database: " + str(final_count))

    finally:
        db.close()

if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    seed_birds()