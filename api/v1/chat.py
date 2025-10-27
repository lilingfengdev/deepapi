"""
OpenAI 兼容的 Chat Completion API
支持 DeepThink 和 UltraThink 模式
"""
import time
import uuid
import json
import logging
import asyncio
from typing import AsyncIterator, Dict, Any, List
from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse

from models import (
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChoice, 
    Message, Usage, MessageContent, extract_text_from_content
)
from config import config
from utils.openai_client import create_client
from utils.summary_think import ThinkingSummaryGenerator, UltraThinkSummaryGenerator, generate_simple_thinking_tag
from engine.deep_think import DeepThinkEngine
from engine.ultra_think import UltraThinkEngine

router = APIRouter()
logger = logging.getLogger(__name__)


def extract_llm_params(request: ChatCompletionRequest) -> Dict[str, Any]:
    """从请求中提取 LLM 参数"""
    params = {}
    
    # 只提取 temperature 和 max_tokens 参数
    if request.temperature is not None:
        params['temperature'] = request.temperature
    if request.max_tokens is not None:
        params['max_tokens'] = request.max_tokens
    
    return params


def process_user_messages(messages: List[Message]) -> List[Message]:
    """
    处理用户发送的消息列表：
    1. 提取所有 system role 消息并合并
    2. 将合并后的 system 消息转为 user 消息
    3. 放在第一条 user role 消息的最前面
    
    Args:
        messages: 原始消息列表
    
    Returns:
        处理后的消息列表（不包含 system role）
    """
    system_messages = []
    non_system_messages = []
    
    # 分离 system 消息和其他消息
    for msg in messages:
        if msg.role == "system":
            system_messages.append(msg)
        else:
            non_system_messages.append(msg)
    
    # 如果没有 system 消息，直接返回原列表
    if not system_messages:
        return messages
    
    # 合并所有 system 消息
    merged_system_content = []
    for msg in system_messages:
        content_text = extract_text_from_content(msg.content)
        if content_text.strip():
            merged_system_content.append(content_text)
    
    # 如果合并后为空，直接返回非 system 消息
    if not merged_system_content:
        return non_system_messages
    
    # 创建转换后的 user 消息（带有明确的标识）
    system_as_user_content = "# System Instructions\n\n" + "\n\n---\n\n".join(merged_system_content)
    system_as_user_msg = Message(
        role="user",
        content=system_as_user_content
    )
    
    # 找到第一条 user 消息的位置
    first_user_index = None
    for i, msg in enumerate(non_system_messages):
        if msg.role == "user":
            first_user_index = i
            break
    
    # 插入转换后的消息
    if first_user_index is not None:
        # 在第一条 user 消息之前插入
        processed_messages = (
            non_system_messages[:first_user_index] +
            [system_as_user_msg] +
            non_system_messages[first_user_index:]
        )
    else:
        # 如果没有 user 消息，放在最前面
        processed_messages = [system_as_user_msg] + non_system_messages
    
    return processed_messages


def verify_auth(authorization: str = Header(None)) -> bool:
    """验证 API 密钥"""
    if not config.api_key:
        return True  # 未设置密钥则不验证
    
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    
    # 支持 "Bearer xxx" 或直接 "xxx"
    token = authorization.replace("Bearer ", "").strip()
    
    if not config.validate_api_key(token):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return True


async def stream_chat_completion(
    request: ChatCompletionRequest,
    model_config,
    provider_config,
) -> AsyncIterator[str]:
    """流式聊天补全"""
    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    engine_task = None  # 用于跟踪引擎任务以便在断开时取消
    
    # 提取 LLM 参数
    llm_params = extract_llm_params(request)
    
    # 保留完整的对话历史
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages found")
    
    # 处理用户消息：将 system role 合并后转为 user 消息
    processed_messages = process_user_messages(request.messages)
    
    # 提取最后一个用户消息作为当前问题
    user_messages = [msg for msg in processed_messages if msg.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")
    
    last_user_message = user_messages[-1]
    # 保留原始的多模态内容（如果有图片）
    problem_statement_raw = last_user_message.content
    # 同时提取纯文本版本用于日志和摘要
    problem_statement_text = extract_text_from_content(problem_statement_raw)
    
    # 构建结构化的对话历史（排除最后一条用户消息）
    conversation_history = []
    if len(processed_messages) > 1:
        context_messages = processed_messages[:-1]  # 排除最后一条消息
        for msg in context_messages:
            conversation_history.append({
                "role": msg.role,
                "content": msg.content
            })
    
    # 创建后端客户端 - 传入RPM限制和重试次数，将在每次调用后端API时进行限流
    max_retry = model_config.get_max_retry(default=config.max_retry)
    client = create_client(provider_config.base_url, provider_config.key, model_config.rpm, max_retry)
    
    # 不再直接注入提示词，而是通过标志传递给引擎
    # 引擎会在正确的时机执行 Ask 和 Plan 阶段
    
    # 如果启用了 summary_think,创建思维链生成器
    thinking_generator = None
    if model_config.has_summary_think:
        if model_config.level == "ultrathink":
            thinking_generator = UltraThinkSummaryGenerator()
        else:
            thinking_generator = ThinkingSummaryGenerator(mode="deepthink")
    
    # 定义进度处理器 - 将进度事件转换为流式输出
    async def stream_progress(event):
        """处理进度事件并流式发送"""
        # 如果启用了 summary_think,将事件转换为思维链
        if thinking_generator:
            thinking_text = thinking_generator.process_event(event)
            if thinking_text:
                # 使用 reasoning_content 字段输出推理过程
                delta = {"reasoning_content": thinking_text}
                chunk_data = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": delta,
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
    
    # 根据模型级别选择引擎
    if model_config.level == "ultrathink":
        # UltraThink 模式
        # 使用生成器来捕获进度并流式输出
        progress_queue = []
        
        def on_progress(event):
            """捕获进度事件"""
            progress_queue.append(event)
        
        def on_agent_update(agent_id: str, update: Dict[str, Any]):
            """捕获 Agent 更新"""
            from models import ProgressEvent
            progress_queue.append(ProgressEvent(
                type="agent-update",
                data={"agentId": agent_id, **update}
            ))
        
        # 运行引擎 - 传递结构化的对话历史和多模态内容
        engine = UltraThinkEngine(
            client=client,
            model=model_config.model,
            problem_statement=problem_statement_raw,  # 传递多模态内容
            conversation_history=conversation_history,  # 传递结构化的消息历史
            max_iterations=model_config.max_iterations,
            required_successful_verifications=model_config.required_verifications,
            num_agents=model_config.num_agent,
            parallel_run_agent=model_config.parallel_run_agent,
            model_stages=model_config.models,
            on_progress=on_progress,
            on_agent_update=on_agent_update,
            enable_parallel_check=model_config.parallel_check,
            llm_params=llm_params,
        )
        
        # 在后台运行引擎
        engine_task = asyncio.create_task(engine.run())
        
        try:
            # 流式发送进度
            while not engine_task.done():
                # 处理队列中的进度事件
                while progress_queue:
                    event = progress_queue.pop(0)
                    async for chunk in stream_progress(event):
                        yield chunk
                await asyncio.sleep(0.1)  # 短暂等待避免busy loop
            
            # 获取最终结果
            result = await engine_task
            
            # 处理剩余的进度事件
            while progress_queue:
                event = progress_queue.pop(0)
                async for chunk in stream_progress(event):
                    yield chunk
            
            # 流式发送最终答案
            final_text = result.summary or result.final_solution
            for i in range(0, len(final_text), 50):
                chunk = final_text[i:i+50]
                delta = {"content": chunk}
                chunk_data = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": delta,
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
        except GeneratorExit:
            # 客户端断开连接，取消引擎任务
            logger.info(f"Client disconnected for request {request_id}, cancelling engine task")
            if engine_task and not engine_task.done():
                engine_task.cancel()
                try:
                    await engine_task
                except asyncio.CancelledError:
                    pass  # 预期的取消异常
            # 不重新抛出 GeneratorExit，让生成器正常结束
        except (asyncio.CancelledError, Exception) as e:
            # 其他异常情况，记录日志并取消任务
            logger.error(f"Error during streaming for request {request_id}: {e}")
            if engine_task and not engine_task.done():
                engine_task.cancel()
                try:
                    await engine_task
                except asyncio.CancelledError:
                    pass
            raise  # 重新抛出异常
    
    else:  # deepthink
        # DeepThink 模式
        progress_queue = []
        
        def on_progress(event):
            """捕获进度事件"""
            progress_queue.append(event)
        
        # 运行引擎 - 传递结构化的对话历史和多模态内容
        engine = DeepThinkEngine(
            client=client,
            model=model_config.model,
            problem_statement=problem_statement_raw,  # 传递多模态内容
            conversation_history=conversation_history,  # 传递结构化的消息历史
            max_iterations=model_config.max_iterations,
            required_successful_verifications=model_config.required_verifications,
            model_stages=model_config.models,
            on_progress=on_progress,
            enable_planning=model_config.has_plan_mode,
            enable_parallel_check=model_config.parallel_check,
            llm_params=llm_params,
        )
        
        # 在后台运行引擎
        engine_task = asyncio.create_task(engine.run())
        
        try:
            # 流式发送进度
            while not engine_task.done():
                # 处理队列中的进度事件
                while progress_queue:
                    event = progress_queue.pop(0)
                    async for chunk in stream_progress(event):
                        yield chunk
                await asyncio.sleep(0.1)  # 短暂等待避免busy loop
            
            # 获取最终结果
            result = await engine_task
            
            # 处理剩余的进度事件
            while progress_queue:
                event = progress_queue.pop(0)
                async for chunk in stream_progress(event):
                    yield chunk
            
            # 流式发送最终答案
            final_text = result.summary or result.final_solution
            for i in range(0, len(final_text), 50):
                chunk = final_text[i:i+50]
                delta = {"content": chunk}
                chunk_data = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": delta,
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
        except GeneratorExit:
            # 客户端断开连接，取消引擎任务
            logger.info(f"Client disconnected for request {request_id}, cancelling engine task")
            if engine_task and not engine_task.done():
                engine_task.cancel()
                try:
                    await engine_task
                except asyncio.CancelledError:
                    pass  # 预期的取消异常
            # 不重新抛出 GeneratorExit，让生成器正常结束
        except (asyncio.CancelledError, Exception) as e:
            # 其他异常情况，记录日志并取消任务
            logger.error(f"Error during streaming for request {request_id}: {e}")
            if engine_task and not engine_task.done():
                engine_task.cancel()
                try:
                    await engine_task
                except asyncio.CancelledError:
                    pass
            raise  # 重新抛出异常
    
    # 发送结束标记
    chunk_data = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(chunk_data)}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: str = Header(None)
):
    """
    OpenAI 兼容的聊天补全端点
    支持 DeepThink、UltraThink 和直接代理模式
    """
    # 验证 API 密钥
    verify_auth(authorization)
    
    # 获取模型配置
    model_config = config.get_model(request.model)
    if not model_config:
        raise HTTPException(status_code=404, detail=f"Model {request.model} not found")
    
    # 获取提供商配置
    provider_config = config.get_provider(model_config.provider)
    if not provider_config:
        raise HTTPException(
            status_code=500,
            detail=f"Provider {model_config.provider} not configured"
        )
    
    # 流式响应
    if request.stream:
        return StreamingResponse(
            stream_chat_completion(request, model_config, provider_config),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
    
    # 非流式响应
    else:
        # 提取 LLM 参数
        llm_params = extract_llm_params(request)
        
        # 保留完整的对话历史
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages found")
        
        # 处理用户消息：将 system role 合并后转为 user 消息
        processed_messages = process_user_messages(request.messages)
        
        # 提取最后一个用户消息作为当前问题
        user_messages = [msg for msg in processed_messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")
        
        last_user_message = user_messages[-1]
        # 保留原始的多模态内容（如果有图片）
        problem_statement_raw = last_user_message.content
        # 同时提取纯文本版本用于日志和摘要
        problem_statement_text = extract_text_from_content(problem_statement_raw)
        
        # 构建结构化的对话历史（排除最后一条用户消息）
        conversation_history = []
        if len(processed_messages) > 1:
            context_messages = processed_messages[:-1]  # 排除最后一条消息
            for msg in context_messages:
                conversation_history.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # 创建后端客户端 - 传入RPM限制和重试次数，将在每次调用后端API时进行限流
        max_retry = model_config.get_max_retry(default=config.max_retry)
        client = create_client(provider_config.base_url, provider_config.key, model_config.rpm, max_retry)
        
        # 根据模型级别处理
        reasoning_text = None
        if model_config.level == "ultrathink":
            engine = UltraThinkEngine(
                client=client,
                model=model_config.model,
                problem_statement=problem_statement_raw,  # 传递多模态内容
                conversation_history=conversation_history,  # 传递结构化的消息历史
                max_iterations=model_config.max_iterations,
                required_successful_verifications=model_config.required_verifications,
                num_agents=model_config.num_agent,
                parallel_run_agent=model_config.parallel_run_agent,
                model_stages=model_config.models,
                enable_parallel_check=model_config.parallel_check,
                llm_params=llm_params,
            )
            result = await engine.run()
            response_text = result.summary or result.final_solution
            
            # 如果启用 summary_think,生成推理内容
            if model_config.has_summary_think:
                reasoning_text = generate_simple_thinking_tag("ultrathink")
        
        else:  # deepthink
            engine = DeepThinkEngine(
                client=client,
                model=model_config.model,
                problem_statement=problem_statement_raw,  # 传递多模态内容
                conversation_history=conversation_history,  # 传递结构化的消息历史
                max_iterations=model_config.max_iterations,
                required_successful_verifications=model_config.required_verifications,
                model_stages=model_config.models,
                enable_planning=model_config.has_plan_mode,
                enable_parallel_check=model_config.parallel_check,
                llm_params=llm_params,
            )
            result = await engine.run()
            response_text = result.summary or result.final_solution
            
            # 如果启用 summary_think,生成推理内容
            if model_config.has_summary_think:
                reasoning_text = generate_simple_thinking_tag("deepthink")
        
        # 构建响应
        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())
        
        return ChatCompletionResponse(
            id=request_id,
            object="chat.completion",
            created=created,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=response_text,
                        reasoning_content=reasoning_text
                    ),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=0,  # 简化处理
                completion_tokens=0,
                total_tokens=0
            )
        )

