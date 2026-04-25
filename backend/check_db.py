from database import SessionLocal
from models.bird import Bird
db = SessionLocal()
count = db.query(Bird).count()
db.close()
print("Birds in database: " + str(count))