from sqlalchemy import Column, Integer, String, Text
from database import Base

class Bird(Base):
    __tablename__ = "birds"

    id              = Column(Integer, primary_key=True, index=True)
    common_name     = Column(String(100), nullable=False)
    scientific_name = Column(String(100))
    family          = Column(String(100))
    habitat         = Column(String(255))
    description     = Column(Text)