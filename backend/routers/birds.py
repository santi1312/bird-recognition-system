from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from database import get_db
from models.bird import Bird
from models.media import Media
from schemas.bird_schema import BirdCreate, BirdResponse, BirdSearchResponse

router = APIRouter()

# Add a new bird
@router.post("/", response_model=BirdResponse)
def create_bird(bird: BirdCreate, db: Session = Depends(get_db)):
    new_bird = Bird(**bird.model_dump())
    db.add(new_bird)
    db.commit()
    db.refresh(new_bird)
    return new_bird

# Get all birds
@router.get("/", response_model=list[BirdResponse])
def get_all_birds(db: Session = Depends(get_db)):
    return db.query(Bird).all()

# Search bird by name
@router.get("/search")
def search_bird(
    name: str = Query(..., min_length=2),
    db: Session = Depends(get_db)
):
    bird = db.query(Bird).filter(
        Bird.common_name.ilike(f"%{name}%")
    ).first()

    if not bird:
        raise HTTPException(status_code=404, detail="Bird not found")

    media = db.query(Media).filter(Media.bird_id == bird.id).all()

    return {
        "bird": bird,
        "media": {
            "images": [m.url for m in media if m.type == "image"],
            "audio" : [m.url for m in media if m.type == "audio"],
            "videos": [m.url for m in media if m.type == "video"],
        }
    }

# Get bird by ID
@router.get("/{bird_id}", response_model=BirdResponse)
def get_bird(bird_id: int, db: Session = Depends(get_db)):
    bird = db.query(Bird).filter(Bird.id == bird_id).first()
    if not bird:
        raise HTTPException(status_code=404, detail="Bird not found")
    return bird