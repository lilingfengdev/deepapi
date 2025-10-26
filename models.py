"""
数据模型定义
定义API请求和响应的数据结构
"""
from typing import Optional, List, Dict, Any, Literal, Union
from pydantic import BaseModel, Field


# ============ OpenAI 兼容的请求/响应模型 ============

class Message(BaseModel):
    """聊天消息"""
    role: Literal["system", "user", "assistant"]
    content: Union[str, List[Dict[str, Any]]]  # 支持文本和多模态内容
    name: Optional[str] = None
    reasoning_content: Optional[str] = None  # OpenAI o1 风格的推理内容


# ============ 多模态内容类型 ============

MessageContent = Union[str, List[Dict[str, Any]]]  # 消息内容可以是文本或多模态


def extract_text_from_content(content: MessageContent) -> str:
    """
    从消息内容中提取文本
    支持纯文本和多模态内容（包含 text 和 image_url）
    """
    if isinstance(content, str):
        return content
    
    # 多模态内容，提取所有文本部分
    text_parts = []
    for item in content:
        if isinstance(item, dict):
            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif item.get("type") == "image_url":
                # 对于图片，添加占位符说明
                image_url = item.get("image_url", {})
                if isinstance(image_url, dict):
                    url = image_url.get("url", "")
                else:
                    url = str(image_url)
                text_parts.append(f"[Image: {url[:50]}...]")
    
    return "\n".join(text_parts) if text_parts else str(content)


class ChatCompletionRequest(BaseModel):
    """聊天补全请求"""
    model: str
    messages: List[Message]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    
    # DeepThink 特定参数
    deep_think_options: Optional[Dict[str, Any]] = None


class ChatCompletionChoice(BaseModel):
    """聊天补全选择"""
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    """Token使用情况"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """聊天补全响应"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[Usage] = None


class ChatCompletionChunk(BaseModel):
    """流式聊天补全块"""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]


# ============ Deep Think 内部模型 ============

class Verification(BaseModel):
    """验证结果"""
    timestamp: int
    passed: bool
    bug_report: str
    good_verify: str


class DeepThinkIteration(BaseModel):
    """Deep Think 迭代"""
    iteration: int
    solution: str
    verification: Verification
    status: Literal["thinking", "verifying", "correcting", "completed", "failed"]


class Source(BaseModel):
    """引用来源"""
    title: Optional[str] = None
    content: Optional[str] = None
    url: str


class DeepThinkResult(BaseModel):
    """Deep Think 结果"""
    mode: Literal["deep-think"] = "deep-think"
    plan: Optional[str] = None
    initial_thought: str
    improvements: List[str] = []
    iterations: List[DeepThinkIteration]
    verifications: List[Verification]
    final_solution: str
    summary: Optional[str] = None
    total_iterations: int
    successful_verifications: int
    sources: Optional[List[Source]] = None
    knowledge_enhanced: bool = False


class AgentResult(BaseModel):
    """Agent 结果"""
    agent_id: str
    approach: str
    specific_prompt: str
    status: Literal["pending", "thinking", "verifying", "completed", "failed"]
    progress: int
    solution: Optional[str] = None
    verifications: Optional[List[Verification]] = None
    error: Optional[str] = None


class UltraThinkResult(BaseModel):
    """Ultra Think 结果"""
    mode: Literal["ultra-think"] = "ultra-think"
    plan: str
    agent_results: List[AgentResult]
    synthesis: str
    final_solution: str
    summary: Optional[str] = None
    total_agents: int
    completed_agents: int
    sources: Optional[List[Source]] = None
    knowledge_enhanced: bool = False


# ============ 进度事件模型 ============

class ProgressEvent(BaseModel):
    """进度事件"""
    type: str
    data: Dict[str, Any]

