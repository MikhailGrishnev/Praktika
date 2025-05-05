from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from database import Base

class DetectionHistory(Base):
    __tablename__ = "detection_history"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    detections = Column(Integer)
    media_type = Column(String)  # image or video
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
