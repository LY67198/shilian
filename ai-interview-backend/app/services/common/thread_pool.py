from concurrent.futures import ThreadPoolExecutor


class ThreadPoolService:
    """线程池服务，用于将同步阻塞操作（如 SMTP 发送）提交到线程池执行，避免阻塞事件循环。"""

    def __init__(self, max_workers: int = 4):
        """初始化线程池。

        Args:
            max_workers: 线程池最大工作线程数，默认 4。
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def get_executor(self):
        """获取线程池实例"""
        return self.executor

    def shutdown(self):
        """关闭线程池"""
        self.executor.shutdown(wait=False)


# 全局单例实例
thread_pool_service = ThreadPoolService()
