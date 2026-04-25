from database import SessionLocal
from models.bird import Bird
from models.media import Media

db = SessionLocal()
db.query(Media).delete()
db.query(Bird).delete()
db.commit()
db.close()
print("Cleared!")