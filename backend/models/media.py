from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship
from database import Base

class Media(Base):
    __tablename__ = "media"

    id      = Column(Integer, primary_key=True, index=True)
    bird_id = Column(Integer, ForeignKey("birds.id"), nullable=False)
    type    = Column(String(10), nullable=False)  # image / audio / video
    url     = Column(Text, nullable=False)
    source  = Column(String(100))

    bird = relationship("Bird", backref="media")