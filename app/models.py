from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from .db import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, default="admin")
    created_at = Column(DateTime, default=datetime.utcnow)

class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True)
    ts = Column(DateTime, default=datetime.utcnow, index=True)
    etype = Column(String, index=True)
    severity = Column(String)
    track_id = Column(Integer)
    a = Column(Integer)
    b = Column(Integer)
    extra = Column(JSON)
