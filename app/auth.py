# app/auth.py
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import jwt, JWTError
from passlib.context import CryptContext

# Use bcrypt_sha256 so we don't hit bcrypt's 72-byte limit
_pwd_ctx = CryptContext(schemes=["bcrypt_sha256", "bcrypt"], deprecated="auto")

JWT_SECRET = os.getenv("JWT_SECRET", "change_me_long_random_secret")
JWT_EXP_MINUTES = int(os.getenv("JWT_EXP_MINUTES", "1440"))
JWT_ALG = "HS256"

# ---------- password hashing ----------
def hash_password(p: str) -> str:
    return _pwd_ctx.hash(p)

def verify_password(p: str, hashed: str) -> bool:
    return _pwd_ctx.verify(p, hashed)

# ---------- token helpers ----------
def create_access_token(claims: Dict[str, Any], minutes: Optional[int] = None) -> str:
    """Create a JWT from a dict of claims."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=minutes or JWT_EXP_MINUTES)
    to_encode = {**claims, "exp": expire}
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALG)

def decode_token(token: str) -> Dict[str, Any]:
    """Decode & verify a JWT, return payload or raise JWTError."""
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])

# Backward-compatible alias
make_token = create_access_token

# ---------- FastAPI dependency to require auth ----------
_bearer = HTTPBearer(auto_error=True)

def require_auth(credentials: HTTPAuthorizationCredentials = Depends(_bearer)) -> Dict[str, Any]:
    """FastAPI dependency that validates Bearer token and returns payload."""
    token = credentials.credentials
    try:
        return decode_token(token)
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
