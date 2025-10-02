# app/db.py
import os
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

# Optional: read YAML if DATABASE_URL isn't in .env
try:
    import yaml  # PyYAML is in requirements.txt
except Exception:  # pragma: no cover
    yaml = None


# ---------- load env ----------
# Loads variables from a .env file in the project root (or any parent dir)
load_dotenv(dotenv_path=Path(".env"))


def _read_url_from_yaml() -> Optional[str]:
    """
    Fallback: read DATABASE_URL from configs/infer.yaml → database.url
    """
    cfg_path = Path("configs") / "infer.yaml"
    if not cfg_path.exists() or yaml is None:
        return None
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return (data.get("database") or {}).get("url")
    except Exception:
        return None


# ---------- resolve DATABASE_URL ----------
DATABASE_URL = os.getenv("DATABASE_URL") or _read_url_from_yaml() or "sqlite:///./accidents.db"

# ---------- engine/session/base ----------
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    # Needed for SQLite when used inside FastAPI (multi-threaded)
    connect_args = {"check_same_thread": False}

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,   # recycle dead connections automatically
    connect_args=connect_args,
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    future=True,
)

Base = declarative_base()


# ---------- FastAPI dependency ----------
def get_db():
    """
    Usage in routes:
        from app.db import get_db
        def endpoint(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# (Optional) quick self-test when running this file directly
if __name__ == "__main__":
    print("DATABASE_URL ->", DATABASE_URL)
    # Lazy import of models so Base.metadata includes them if you run this script
    try:
        from app import models  # noqa: F401
        Base.metadata.create_all(bind=engine)
        print("Tables created (if not existed).")
    except Exception as e:
        print("Self-test note:", e)
