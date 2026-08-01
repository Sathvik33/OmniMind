from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from backend.app.api.deps import get_current_user
from backend.app.core.security import create_access_token, hash_password, verify_password
from backend.app.db.database import get_db
from backend.app.db.models import User

router = APIRouter(prefix="/auth", tags=["auth"])

_EMAIL_LOCAL = re.compile(r"^[^@]+")


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)


class UserOut(BaseModel):
    id: int
    email: str
    username: str

    class Config:
        from_attributes = True


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


def _username_from_email(email: str) -> str:
    local = _EMAIL_LOCAL.match(email.lower())
    base = (local.group(0) if local else "user")[:40]
    base = re.sub(r"[^a-z0-9._-]", "", base) or "user"
    return base


def _unique_username(db: Session, email: str) -> str:
    base = _username_from_email(email)
    candidate = base
    n = 1
    while db.query(User).filter(User.username == candidate).first():
        candidate = f"{base}{n}"
        n += 1
    return candidate


@router.post("/signup", response_model=AuthResponse)
def signup(body: SignupRequest, db: Session = Depends(get_db)):
    email = body.email.lower().strip()
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        email=email,
        username=_unique_username(db, email),
        password_hash=hash_password(body.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(str(user.id), extra={"email": user.email})
    return AuthResponse(
        access_token=token,
        user=UserOut(id=user.id, email=user.email or email, username=user.username),
    )


@router.post("/login", response_model=AuthResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    email = body.email.lower().strip()
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(body.password, user.password_hash or ""):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    token = create_access_token(str(user.id), extra={"email": user.email})
    return AuthResponse(
        access_token=token,
        user=UserOut(id=user.id, email=user.email or email, username=user.username),
    )


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return UserOut(id=user.id, email=user.email or "", username=user.username)
