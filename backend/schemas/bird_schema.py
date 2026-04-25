from pydantic import BaseModel
from typing import Optional, List

class BirdBase(BaseModel):
    common_name     : str
    scientific_name : Optional[str] = None
    family          : Optional[str] = None
    habitat         : Optional[str] = None
    description     : Optional[str] = None

class BirdCreate(BirdBase):
    pass

class BirdResponse(BirdBase):
    id: int

    class Config:
        from_attributes = True

class MediaResponse(BaseModel):
    id      : int
    bird_id : int
    type    : str
    url     : str
    source  : Optional[str] = None

    class Config:
        from_attributes = True

class BirdSearchResponse(BaseModel):
    bird  : BirdResponse
    media : dict