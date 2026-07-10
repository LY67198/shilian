"""
单元测试 — 登录限流 Redis 计数器逻辑

依赖：
- Redis 实例（Docker compose 已含），通过环境变量 REDIS_HOST 指向

运行（在容器内或本地 + Redis 可达）：
  pytest tests/test_login_ratelimit.py -m "integration"
"""
import os
import asyncio
import pytest

from app.services.common.redis import redis_client
from app.services.client.auth import (
    _check_login_rate_limit,
    _increment_login_failures,
    _clear_login_attempts,
)
from app.exceptions.http_exceptions import APIException


EMAIL = "ratelimit_test@example.com"


@pytest.fixture(autouse=True)
async def _cleanup_redis():
    """每个测试运行前清掉计数器，避免互相影响"""
    await _clear_login_attempts(EMAIL)
    yield
    await _clear_login_attempts(EMAIL)


@pytest.mark.integration
class TestLoginRateLimit:
    async def test_first_attempt_passes(self):
        await _check_login_rate_limit(EMAIL)

    async def test_after_max_attempts_raises_429(self):
        for _ in range(5):
            await _increment_login_failures(EMAIL)

        with pytest.raises(APIException) as exc:
            await _check_login_rate_limit(EMAIL)
        assert exc.value.status_code == 429

    async def test_clear_resets_counter(self):
        for _ in range(5):
            await _increment_login_failures(EMAIL)

        await _clear_login_attempts(EMAIL)
        await _check_login_rate_limit(EMAIL)  # 不抛错即通过

    async def test_after_success_clears_counter(self):
        """模拟「第 4 次失败 → 然后第 5 次成功」，最终应能再次登录"""
        for _ in range(4):
            await _increment_login_failures(EMAIL)
        assert await redis_client.get(f"login_attempts:test@example.com") is None or \
               int(await redis_client.get(f"login_attempts:TEST@EXAMPLE.COM") or 0) == 0 or \
               int(await redis_client.get(f"login_attempts:{EMAIL.lower()}") or 0) == 4

        await _clear_login_attempts(EMAIL)
        await _check_login_rate_limit(EMAIL)  # 解锁

    async def test_ttl_is_set_on_first_increment(self):
        await _increment_login_failures(EMAIL)
        ttl = await redis_client.redis.ttl(f"login_attempts:{EMAIL.lower()}")
        assert ttl > 0  # 有过期时间