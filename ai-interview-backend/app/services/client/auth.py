from typing import Optional, Dict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from datetime import datetime, timedelta, UTC
from app.models.user import User
from app.models.token import Token
from app.core.security import AuthBase
from app.core.config import settings
from app.db.session import transaction
from app.exceptions.http_exceptions import APIException
from app.services.client.redis_verification import redis_verification_service
from app.schedule.jobs.email_tasks import send_verification_email_task
from app.services.common.redis import redis_client
import re

import logging

_login_logger = logging.getLogger(__name__)


async def _check_login_rate_limit(email: str) -> None:
    """登录前从 Redis 检查失败计数器，超阈值拒绝"""
    key = f"login_attempts:{email.lower()}"
    attempts = await redis_client.get(key)
    max_attempts = settings.LOGIN_RATE_LIMIT_MAX_ATTEMPTS
    if attempts and int(attempts) >= max_attempts:
        ttl = settings.LOGIN_RATE_LIMIT_WINDOW_SECONDS
        raise APIException(
            status_code=429,
            message=f"登录尝试过于频繁，请等待约 {ttl // 60} 分钟后再试",
        )


async def _increment_login_failures(email: str) -> None:
    """登录失败时增加计数器，首次自动带滑动窗口 TTL"""
    key = f"login_attempts:{email.lower()}"
    count = await redis_client.incr_with_ttl(key, settings.LOGIN_RATE_LIMIT_WINDOW_SECONDS)
    _login_logger.warning(f"[login ratelimit] {email} 当前失败计数 {count}")


async def _clear_login_attempts(email: str) -> None:
    """登录成功后清除失败计数（best-effort，Redis 异常不影响主登录流程）"""
    try:
        await redis_client.delete(f"login_attempts:{email.lower()}")
    except Exception:
        _login_logger.debug(f"[login ratelimit] 清除计数器失败（忽略）")


class ClientAuthService(AuthBase):
    def validate_password(self, password: str) -> bool:
        """验证密码强度"""
        if len(password) < 8:
            raise APIException(status_code=400, message="密码至少8个字符")

        if not re.search(r'[A-Za-z]', password):
            raise APIException(status_code=400, message="密码必须包含至少一个字母")

        if not re.search(r'\d', password):
            raise APIException(status_code=400, message="密码必须包含至少一个数字")

        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            raise APIException(status_code=400, message="密码必须包含至少一个特殊字符")

        return True

    async def authenticate_user(self, db: AsyncSession, email: str, password: str) -> Optional[User]:
        """用户身份验证"""
        user_query = select(User).where(User.email == email)
        result = await db.execute(user_query)
        user = result.scalar_one_or_none()

        if not user or not user.verify_password(password):
            return None
        return user

    async def register_user(self, db: AsyncSession, user_data: Dict) -> Dict:
        """用户注册（信息采集）"""
        async with transaction(db):
            # 检查邮箱是否已存在
            existing_user = await db.execute(
                select(User).where(User.email == user_data["email"])
            )
            if existing_user.scalar_one_or_none():
                raise APIException(status_code=400, message="该邮箱已被注册")

            # 验证密码强度
            self.validate_password(user_data["password"])

            # 创建用户（未验证状态）
            user = User(
                email=user_data["email"],
                hashed_password=User.get_password_hash(user_data["password"]),
                first_name=user_data.get("first_name"),
                last_name=user_data.get("last_name"),
                phone=user_data.get("phone"),
                phone_country_code=user_data.get("phone_country_code"),
                university=user_data.get("university"),
                career_goal=user_data.get("career_goal"),
                contract_types=user_data.get("contract_types"),
                location=user_data.get("location"),
                is_active=True,
                is_verified=False
            )

            db.add(user)
            await db.flush()

            # 发送验证码
            await redis_verification_service.check_send_rate_limit(user_data["email"])
            code = await redis_verification_service.generate_and_store_code(
                user_data["email"], "registration", user.id
            )

            # 异步发送邮件
            send_verification_email_task.delay(
                user_data["email"],
                code,
                "registration",
                user_data.get("first_name")
            )

            return {
                "user_id": user.id,
                "email": user.email,
                "message": "注册成功，请验证邮箱",
                "verification_required": True
            }

    async def send_verification_code(self, db: AsyncSession, email: str, code_type: str) -> Dict:
        """发送验证码"""
        # 检查用户是否存在
        user_query = select(User).where(User.email == email)
        result = await db.execute(user_query)
        user = result.scalar_one_or_none()

        if code_type == "registration" and not user:
            raise APIException(status_code=404, message="用户不存在")

        if code_type == "password-reset" and not user:
            raise APIException(status_code=404, message="该邮箱未注册")

        # 检查发送频率限制
        await redis_verification_service.check_send_rate_limit(email)

        # 生成并存储验证码
        code = await redis_verification_service.generate_and_store_code(
            email, code_type, user.id if user else None
        )

        # 异步发送邮件
        send_verification_email_task.delay(
            email,
            code,
            code_type,
            user.first_name if user else None
        )

        return {
            "message": "验证码已发送至您的邮箱",
            "expires_in": 600,  # 10分钟
            "can_resend_at": 60  # 1分钟后可重新发送
        }

    async def verify_email_and_login(self, db: AsyncSession, email: str, code: str) -> Dict:
        """验证邮箱并自动登录"""
        # 验证验证码
        await redis_verification_service.verify_code(email, code, "registration")

        async with transaction(db):
            # 获取用户
            user = await db.execute(select(User).where(User.email == email))
            user = user.scalar_one_or_none()

            if not user:
                raise APIException(status_code=404, message="用户不存在")

            # 标记邮箱已验证
            user.is_verified = True
            user.email_verified_at = datetime.now(UTC)

            # 清除该用户的所有旧令牌
            await db.execute(
                update(Token).where(
                    (Token.user_id == user.id) &
                    (Token.is_active == True)
                ).values(is_active=False)
            )

            # 生成令牌并登录
            access_token = AuthBase.create_access_token(
                str(user.id),
                scope="client",
                expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
            )
            refresh_token = AuthBase.create_refresh_token(str(user.id))

            # 存储 refresh token
            hashed_token = AuthBase.hash_token(refresh_token)
            token = Token(
                user_id=user.id,
                token=hashed_token,
                expires_at=datetime.now(UTC) + timedelta(days=7),
                is_active=True
            )
            db.add(token)

            return {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer",
                "user": {
                    "id": user.id,
                    "email": user.email,
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                    "is_verified": True,
                    "university": user.university,
                    "career_goal": user.career_goal,
                    "location": user.location
                }
            }

    async def login(self, db: AsyncSession, email: str, password: str, remember_me: bool = False) -> Dict:
        """用户登录"""
        # 登录限流：超过阈值直接拒绝，避免暴力破解
        await _check_login_rate_limit(email)

        async with transaction(db):
            user = await self.authenticate_user(db, email, password)
            if not user:
                # 失败时增加计数，便于限流判断
                await _increment_login_failures(email)
                raise APIException(status_code=400, message="邮箱或密码错误")

            if not user.is_active:
                raise APIException(status_code=400, message="账号已被禁用")

            # 登录成功，清除该邮箱的失败计数
            await _clear_login_attempts(email)

            # 清除该用户的所有旧令牌
            await db.execute(
                update(Token).where(
                    (Token.user_id == user.id) &
                    (Token.is_active == True)
                ).values(is_active=False)
            )

            # 生成新的 access token 和 refresh token
            access_token = AuthBase.create_access_token(
                str(user.id),
                scope="client",
                expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
            )

            # 根据 remember_me 设置 refresh token 过期时间
            refresh_expires = timedelta(days=30 if remember_me else 7)
            refresh_token = AuthBase.create_refresh_token(
                str(user.id),
                expires_delta=refresh_expires
            )

            # 存储新的refresh token
            hashed_token = AuthBase.hash_token(refresh_token)
            token = Token(
                user_id=user.id,
                token=hashed_token,
                expires_at=datetime.now(UTC) + refresh_expires,
                is_active=True
            )
            db.add(token)

            # 更新最后活跃时间
            user.last_active_at = datetime.now(UTC)

            return {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer",
                "user": {
                    "id": user.id,
                    "email": user.email,
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                    "is_verified": user.is_verified,
                    "university": user.university,
                    "career_goal": user.career_goal,
                    "location": user.location
                }
            }

    async def refresh_token(self, db: AsyncSession, refresh_token: str) -> Dict:
        """刷新用户令牌"""
        payload = AuthBase.verify_token(refresh_token, scope="refresh")
        if not payload:
            raise APIException(status_code=401, message="无效的刷新令牌")

        user_id = payload.get("sub")
        token_query = select(Token).where(
            (Token.user_id == user_id) &
            (Token.is_active == True)
        )
        result = await db.execute(token_query)
        token = result.scalar_one_or_none()

        if not token or not AuthBase.verify_token_hash(refresh_token, token.token):
            raise APIException(status_code=401, message="无效或已过期的刷新令牌")

        # 检查用户状态
        user = await db.execute(select(User).where(User.id == user_id))
        user = user.scalar_one_or_none()

        if not user or not user.is_active or not user.is_verified:
            raise APIException(status_code=401, message="用户状态异常")

        # 更新令牌使用时间
        token.last_used_at = datetime.now(UTC)
        await db.commit()

        # 生成新的 access token
        access_token = AuthBase.create_access_token(
            user_id,
            scope="client",
            expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        )

        return {
            "access_token": access_token,
            "token_type": "bearer"
        }

    async def logout(self, db: AsyncSession, refresh_token: str) -> None:
        """用户登出"""
        payload = AuthBase.verify_token(refresh_token, scope="refresh")
        if not payload:
            return  # 忽略无效令牌

        user_id = payload.get("sub")
        token_query = select(Token).where(
            (Token.user_id == user_id) &
            (Token.is_active == True)
        )
        result = await db.execute(token_query)
        token = result.scalar_one_or_none()

        if token and AuthBase.verify_token_hash(refresh_token, token.token):
            token.is_active = False
            await db.commit()

    async def verify_password_reset_code(self, db: AsyncSession, email: str, code: str) -> str:
        """验证密码重置验证码"""
        # 验证验证码
        await redis_verification_service.verify_code(email, code, "password-reset")

        # 检查用户是否存在
        user = await db.execute(select(User).where(User.email == email))
        user = user.scalar_one_or_none()

        if not user:
            raise APIException(status_code=404, message="用户不存在")

        # 生成密码重置令牌（短期有效）
        reset_token = AuthBase.create_access_token(
            str(user.id),
            scope="password-reset",
            expires_delta=timedelta(minutes=15)  # 15分钟有效
        )

        return reset_token

    async def reset_password(self, db: AsyncSession, reset_token: str, new_password: str) -> None:
        """重置密码"""
        payload = AuthBase.verify_token(reset_token, scope="password-reset")
        if not payload:
            raise APIException(status_code=401, message="无效的重置令牌")

        # 验证新密码强度
        self.validate_password(new_password)

        user_id = payload.get("sub")
        async with transaction(db):
            # 获取用户
            user = await db.execute(select(User).where(User.id == user_id))
            user = user.scalar_one_or_none()

            if not user:
                raise APIException(status_code=404, message="用户不存在")

            # 更新密码
            user.hashed_password = User.get_password_hash(new_password)

            # 清除所有现有令牌（强制重新登录）
            await db.execute(
                update(Token).where(
                    (Token.user_id == user.id) &
                    (Token.is_active == True)
                ).values(is_active=False)
            )


client_auth_service = ClientAuthService()
