"""
OpenAI 客户端包装器
用于调用后端 LLM 提供商
"""
from typing import Optional, Dict, Any, List, AsyncIterator
from openai import AsyncOpenAI
import json
import logging
import httpx

logger = logging.getLogger(__name__)


class OpenAIClient:
    """OpenAI 客户端包装器"""
    
    def __init__(self, base_url: str, api_key: str, rpm: Optional[int] = None):
        # 创建带有超时设置的 HTTP 客户端
        # 这样可以确保在客户端断开时，HTTP 请求能更快响应取消
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(3000.0, connect=10.0, read=1200.0),  # 总超时300秒，读取120秒
        )
        
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=http_client,
        )
        self.rpm = rpm
        self.rate_limiter = None
        
        # 统计信息
        self.api_calls = 0
        self.total_tokens = 0
        
        # 如果设置了RPM限制，导入限流器
        if rpm:
            from utils.rate_limiter import rate_limiter
            self.rate_limiter = rate_limiter
    
    def get_statistics(self) -> Dict[str, int]:
        """获取统计信息"""
        return {
            "api_calls": self.api_calls,
            "total_tokens": self.total_tokens
        }
    
    async def generate_text(
        self,
        model: str,
        prompt: str = None,
        messages: List[Dict[str, Any]] = None,
        system: str = None,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        生成文本
        支持多模态内容（文本和图片）
        
        Args:
            model: 模型名称
            prompt: 提示词(如果不使用messages，可以是字符串或多模态内容)
            messages: 消息列表（支持多模态content）
            system: 系统提示词
            temperature: 温度参数
            max_tokens: 最大token数
        
        Returns:
            生成的文本
        """
        # RPM限制 - 在调用后端API之前等待
        if self.rate_limiter and self.rpm:
            await self.rate_limiter.wait_for_rate_limit(
                f"backend_api_{model}",
                self.rpm,
                60
            )
        
        # 构建消息列表
        if messages is None:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            if prompt:
                # prompt 可以是字符串或多模态内容
                messages.append({"role": "user", "content": prompt})
        else:
            # 如果提供了system,插入到消息列表开头
            if system:
                messages = [{"role": "system", "content": system}] + messages
        
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # 统计 API 调用
        self.api_calls += 1
        if hasattr(response, 'usage') and response.usage:
            self.total_tokens += response.usage.total_tokens
        
        # 检查响应是否包含 choices
        if not response.choices or len(response.choices) == 0:
            logger.error(f"API 返回空响应: model={model}, response={response}")
            raise ValueError(f"API 返回空响应，可能是模型过载或请求被拒绝。模型: {model}")
        
        return response.choices[0].message.content
    
    async def generate_object(
        self,
        model: str,
        prompt: str,
        response_format: Dict[str, Any],
        temperature: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成结构化对象(JSON)
        
        Args:
            model: 模型名称
            prompt: 提示词
            response_format: 响应格式定义
            temperature: 温度参数
        
        Returns:
            解析后的JSON对象
        """
        # RPM限制 - 在调用后端API之前等待
        if self.rate_limiter and self.rpm:
            await self.rate_limiter.wait_for_rate_limit(
                f"backend_api_{model}",
                self.rpm,
                60
            )
        
        response = await self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format={"type": "json_object"},
            **kwargs
        )
        
        # 统计 API 调用
        self.api_calls += 1
        if hasattr(response, 'usage') and response.usage:
            self.total_tokens += response.usage.total_tokens
        
        # 检查响应是否包含 choices
        if not response.choices or len(response.choices) == 0:
            logger.error(f"API 返回空响应: model={model}, response={response}")
            raise ValueError(f"API 返回空响应，可能是模型过载或请求被拒绝。模型: {model}")
        
        text = response.choices[0].message.content
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # 尝试从代码块中提取JSON
            if "```json" in text:
                json_text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                json_text = text.split("```")[1].split("```")[0].strip()
            else:
                json_text = text.strip()
            
            return json.loads(json_text)
    
    async def stream_text(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        流式生成文本
        支持多模态内容（文本和图片）
        
        Args:
            model: 模型名称
            messages: 消息列表（支持多模态content）
            temperature: 温度参数
            max_tokens: 最大token数
        
        Yields:
            文本块
        """
        # RPM限制 - 在调用后端API之前等待
        if self.rate_limiter and self.rpm:
            await self.rate_limiter.wait_for_rate_limit(
                f"backend_api_{model}",
                self.rpm,
                60
            )
        
        stream = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )
        
        # 统计 API 调用
        self.api_calls += 1
        
        async for chunk in stream:
            # 检查 chunk 是否包含 choices
            if not chunk.choices or len(chunk.choices) == 0:
                logger.warning(f"流式响应中收到空 chunk: model={model}")
                continue
            
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


def create_client(base_url: str, api_key: str, rpm: Optional[int] = None) -> OpenAIClient:
    """创建OpenAI客户端"""
    return OpenAIClient(base_url, api_key, rpm)

