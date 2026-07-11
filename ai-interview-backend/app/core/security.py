import hashlib
import hmac
import uuid
from datetime import datetime, timedelta, UTC
from typing import Optional, Dict

from jose import jwt, JWTError
from passlib.context import CryptContext

from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def create_access_token(
    subject: str,
    scope: str,
    expires_delta: Optional[timedelta] = None
) -> str:
    """创建访问令牌"""
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode = {
        "exp": expire,
        "sub": str(subject),
        "scope": scope,
        "jti": str(uuid.uuid4()),
    }
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_refresh_token(subject: str, expires_delta: Optional[timedelta] = None) -> str:
    """创建刷新令牌"""
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(days=7)

    to_encode = {
        "exp": expire,
        "sub": str(subject),
        "jti": str(uuid.uuid4()),
        "scope": "refresh",
    }
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def verify_token(token: str, scope: str = None) -> Optional[Dict]:
    """验证令牌"""
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        if scope and payload.get("scope") != scope:
            return None
        return payload
    except JWTError:
        return None


# ── Token 哈希（SHA-256，无长度限制）──


def hash_token(token: str) -> str:
    """对 JWT token 进行 SHA-256 哈希"""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def verify_token_hash(plain_token: str, hashed_token: str) -> bool:
    """验证 token 哈希（恒定时间比较）"""
    return hmac.compare_digest(
        hashlib.sha256(plain_token.encode("utf-8")).hexdigest(),
        hashed_token,
    )


# ── 密码哈希（bcrypt）──


def get_password_hash(password: str) -> str:
    """密码 bcrypt 哈希"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    return pwd_context.verify(plain_password, hashed_password)
