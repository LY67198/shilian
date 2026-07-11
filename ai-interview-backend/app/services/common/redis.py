from redis.asyncio import Redis
from app.core.config import settings


class RedisClient:
    """Redis 异步客户端封装，提供键值存储、冷却检查和管道操作。"""

    def __init__(self):
        """初始化 Redis 连接，使用配置文件中的主机、端口和密码参数。"""
        # 构建Redis连接参数
        redis_params = {
            "host": settings.REDIS_HOST,
            "port": settings.REDIS_PORT,
            "decode_responses": True
        }
        
        # 只有当密码不为空时才添加密码参数
        if hasattr(settings, "REDIS_PASSWORD") and settings.REDIS_PASSWORD:
            redis_params["password"] = settings.REDIS_PASSWORD
            
        self.redis = Redis(**redis_params)

    async def set_with_ttl(self, key: str, value: str, ttl_seconds: int):
        """设置键值对，带过期时间"""
        await self.redis.setex(key, ttl_seconds, value)

    async def get(self, key: str) -> str:
        """获取值"""
        return await self.redis.get(key)

    async def delete(self, key: str):
        """删除键"""
        await self.redis.delete(key)

    async def incr_with_ttl(self, key: str, ttl_seconds: int) -> int:
        """自增并在首次设置时带过期时间，返回自增后的值"""
        pipe = self.redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, ttl_seconds, nx=True)
        results = await pipe.execute()
        return results[0]

    async def set_cooldown(self, key: str, ttl_seconds: int):
        """设置冷却时间"""
        await self.redis.setex(key, ttl_seconds, "1")

    async def check_cooldown(self, key: str) -> bool:
        """检查是否在冷却中"""
        return bool(await self.redis.exists(key))

    def pipeline(self, *args, **kwargs):
        """
        兼容原生redis-py的pipeline用法，直接转发到底层redis实例。
        注意：如果底层用的是redis.asyncio.Redis，返回的是异步pipeline。
        """
        return self.redis.pipeline(*args, **kwargs)

    async def brpop(self, key, timeout=1):
        """阻塞式从列表右侧弹出元素。

        Args:
            key: Redis 键名。
            timeout: 阻塞超时时间（秒），默认 1 秒。

        Returns:
            弹出元素组成的元组，超时返回 None。
        """
        return await self.redis.brpop(key, timeout=timeout)

    async def close(self):
        """关闭Redis连接"""
        await self.redis.close()


redis_client = RedisClient()
