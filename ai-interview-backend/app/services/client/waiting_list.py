from typing import Dict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timedelta, UTC
from app.models.waiting_list import WaitingList
from app.core.security import verify_token as _verify_jwt
from app.core.config import settings
from app.db.session import transaction
from app.exceptions.http_exceptions import APIException
from app.services.common.redis import redis_client
from app.schedule.jobs.email_tasks import send_waiting_list_verification_task, send_waiting_list_admin_notification_task


class WaitingListService:
    """等待列表服务，提供邮箱验证令牌生成/校验、IP 限流、申请提交和重发验证邮件等功能。"""

    async def generate_verification_token(self, email: str, waiting_list_id: int) -> str:
        """生成 JWT 邮箱验证令牌，并将唯一 jti 存入 Redis。

        Args:
            email: 用户邮箱。
            waiting_list_id: 等待列表记录 ID。

        Returns:
            JWT 验证令牌字符串，有效期 24 小时。
        """
        from jose import jwt
        from datetime import datetime, UTC
        import uuid

        expire = datetime.now(UTC) + timedelta(hours=24)
        jti = str(uuid.uuid4())  # 唯一令牌标识符

        to_encode = {
            "sub": str(waiting_list_id),
            "email": email,
            "scope": "waiting-list-verification",
            "exp": expire,
            "jti": jti
        }

        token = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

        # 将当前有效令牌存储到 Redis
        redis_key = f"waiting_list:token:{email}"
        await redis_client.set_with_ttl(redis_key, jti, 24 * 3600)  # 24小时

        return token

    async def verify_token(self, token: str) -> Dict:
        """校验 JWT 验证令牌的有效性。

        Args:
            token: JWT 验证令牌。

        Returns:
            解析后的 JWT payload 字典。

        Raises:
            APIException: 令牌无效或已被更新的链接替代。
        """
        payload = _verify_jwt(token, scope="waiting-list-verification")
        if not payload:
            raise APIException(status_code=400, message="无效或已过期的验证链接")

        # 检查此令牌是否仍为该邮箱的当前有效令牌
        email = payload.get("email")
        jti = payload.get("jti")

        if email and jti:
            redis_key = f"waiting_list:token:{email}"
            current_jti = await redis_client.get(redis_key)

            if current_jti != jti:
                raise APIException(status_code=400, message="此验证链接已被更新的链接替代")

        return payload

    async def check_ip_rate_limit(self, ip_address: str) -> None:
        """检查 IP 地址的申请频率限制（每小时最多 100 次）。

        Args:
            ip_address: 客户端 IP 地址。

        Raises:
            APIException: IP 请求次数超过限制。
        """
        redis_key = f"waiting_list:ip:{ip_address}"
        count_str = await redis_client.get(redis_key)

        if count_str and int(count_str) >= 100:
            raise APIException(
                status_code=400,
                message="该 IP 地址请求过多，请1小时后再试"
            )

        if count_str:
            await redis_client.redis.incr(redis_key)
        else:
            await redis_client.set_with_ttl(redis_key, "1", 3600)

    async def submit_application(
        self,
        db: AsyncSession,
        data: Dict,
        ip_address: str,
        user_agent: str
    ) -> Dict:
        """提交等待列表申请，发送邮箱验证邮件。

        若邮箱已存在且已验证则拒绝；若已存在但未验证则重发验证邮件。

        Args:
            db: 数据库会话。
            data: 申请数据，包含 first_name、last_name、email、university。
            ip_address: 客户端 IP 地址。
            user_agent: 客户端 User-Agent。

        Returns:
            包含 email 和 message 的字典。
        """
        existing_query = select(WaitingList).where(WaitingList.email == data["email"])
        result = await db.execute(existing_query)
        existing = result.scalar_one_or_none()

        if existing and existing.is_verified:
            raise APIException(
                status_code=400,
                message="该邮箱已在等待列表中"
            )

        await self.check_ip_rate_limit(ip_address)

        if existing and not existing.is_verified:
            token = await self.generate_verification_token(
                data["email"],
                existing.id
            )
            send_waiting_list_verification_task.delay(
                data["email"],
                token,
                data["first_name"]
            )
            return {
                "email": data["email"],
                "message": "验证邮件已重新发送，请查收邮箱"
            }

        async with transaction(db):
            new_record = WaitingList(
                first_name=data["first_name"],
                last_name=data["last_name"],
                email=data["email"],
                university=data.get("university"),
                ip_address=ip_address,
                user_agent=user_agent,
                is_verified=False
            )

            db.add(new_record)
            await db.flush()

            token = await self.generate_verification_token(
                data["email"],
                new_record.id
            )

            send_waiting_list_verification_task.delay(
                data["email"],
                token,
                data["first_name"]
            )

            return {
                "email": data["email"],
                "message": "请查收邮箱进行验证"
            }

    async def verify_email(self, db: AsyncSession, token: str) -> Dict:
        """验证邮箱并标记等待列表记录为已验证，同时通知管理员。

        Args:
            db: 数据库会话。
            token: JWT 验证令牌。

        Returns:
            包含 message、email 和 verified_at 的字典。
        """
        payload = await self.verify_token(token)
        waiting_list_id = int(payload.get("sub"))
        email = payload.get("email")

        async with transaction(db):
            query = select(WaitingList).where(WaitingList.id == waiting_list_id)
            result = await db.execute(query)
            record = result.scalar_one_or_none()

            if not record:
                raise APIException(status_code=404, message="等待列表记录不存在")

            if record.email != email:
                raise APIException(status_code=400, message="无效的验证链接")

            if record.is_verified:
                raise APIException(status_code=400, message="邮箱已验证")

            record.is_verified = True
            record.verified_at = datetime.now(UTC)

            # 从 Redis 清除验证令牌
            redis_key = f"waiting_list:token:{email}"
            await redis_client.delete(redis_key)

            send_waiting_list_admin_notification_task.delay({
                "first_name": record.first_name,
                "last_name": record.last_name,
                "email": record.email,
                "university": record.university,
                "verified_at": record.verified_at.isoformat()
            })

            return {
                "message": "邮箱验证成功",
                "email": record.email,
                "verified_at": record.verified_at
            }

    async def resend_verification(self, db: AsyncSession, email: str, ip_address: str) -> Dict:
        """重新发送邮箱验证邮件。

        Args:
            db: 数据库会话。
            email: 用户邮箱。
            ip_address: 客户端 IP 地址。

        Returns:
            包含 message 和 email 的字典。
        """
        query = select(WaitingList).where(WaitingList.email == email)
        result = await db.execute(query)
        record = result.scalar_one_or_none()

        if not record:
            raise APIException(status_code=404, message="该邮箱不在等待列表中")

        if record.is_verified:
            raise APIException(status_code=400, message="该邮箱已验证")

        await self.check_ip_rate_limit(ip_address)

        token = await self.generate_verification_token(email, record.id)

        send_waiting_list_verification_task.delay(
            email,
            token,
            record.first_name
        )

        return {
            "message": "验证邮件已重新发送",
            "email": email
        }


waiting_list_service = WaitingListService()
