"""
速率限制器
实现每分钟请求数(RPM)限制
用于限制后端调用LLM API的频率，而非限制用户请求频率
使用 aiolimiter 库实现
"""
from typing import Dict
from aiolimiter import AsyncLimiter


class RateLimiterManager:
    """速率限制器管理器，为每个key维护独立的限流器"""
    
    def __init__(self):
        self._limiters: Dict[str, AsyncLimiter] = {}
    
    def _get_limiter(self, key: str, limit: int, window: int = 60) -> AsyncLimiter:
        """获取或创建指定key的限流器"""
        limiter_key = f"{key}:{limit}:{window}"
        
        if limiter_key not in self._limiters:
            # 创建新的限流器: max_rate 次/time_period 秒
            self._limiters[limiter_key] = AsyncLimiter(max_rate=limit, time_period=window)
        
        return self._limiters[limiter_key]
    
    async def wait_for_rate_limit(self, key: str, limit: int, window: int = 60):
        """
        等待直到可以发起请求（自动获取令牌）
        
        Args:
            key: 限制的键(通常是后端API模型ID)
            limit: 限制数量
            window: 时间窗口(秒),默认60秒
        """
        limiter = self._get_limiter(key, limit, window)
        async with limiter:
            # 自动等待直到有可用配额
            pass
    
    def has_capacity(self, key: str, limit: int, window: int = 60) -> bool:
        """
        检查是否有可用配额（非阻塞）
        
        Args:
            key: 限制的键
            limit: 限制数量
            window: 时间窗口(秒)
        
        Returns:
            True如果有可用配额，False如果需要等待
        """
        limiter = self._get_limiter(key, limit, window)
        return limiter.has_capacity()


# 全局速率限制器实例
rate_limiter = RateLimiterManager()

