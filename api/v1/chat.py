"""
OpenAI 兼容的 Chat Completion API
支持 DeepThink 和 UltraThink 模式
"""
import time
import uuid
import json
from typing import AsyncIterator, Dict, Any
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


def extract_llm_params(request: ChatCompletionRequest) -> Dict[str, Any]:
    """从请求中提取 LLM 参数"""
    params = {}
    
    # 提取标准 OpenAI 参数
    if request.temperature is not None:
        params['temperature'] = request.temperature
    if request.top_p is not None:
        params['top_p'] = request.top_p
    if request.max_tokens is not None:
        params['max_tokens'] = request.max_tokens
    if request.presence_penalty is not None:
        params['presence_penalty'] = request.presence_penalty
    if request.frequency_penalty is not None:
        params['frequency_penalty'] = request.frequency_penalty
    if request.stop is not None:
        params['stop'] = request.stop
    
    return params


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
    
    # 提取 LLM 参数
    llm_params = extract_llm_params(request)
    
    # 保留完整的对话历史
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages found")
    
    # 提取最后一个用户消息作为当前问题
    user_messages = [msg for msg in request.messages if msg.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")
    
    last_user_message = user_messages[-1]
    # 保留原始的多模态内容（如果有图片）
    problem_statement_raw = last_user_message.content
    # 同时提取纯文本版本用于日志和摘要
    problem_statement_text = extract_text_from_content(problem_statement_raw)
    
    # 构建完整的对话上下文（排除最后一条用户消息）
    conversation_context = ""
    if len(request.messages) > 1:
        context_messages = request.messages[:-1]  # 排除最后一条消息
        conversation_parts = []
        for msg in context_messages:
            # 提取文本用于上下文说明
            content_text = extract_text_from_content(msg.content)
            conversation_parts.append(f"{msg.role.upper()}: {content_text}")
        conversation_context = "\n\n".join(conversation_parts)
    
    # 创建后端客户端 - 传入RPM限制，将在每次调用后端API时进行限流
    client = create_client(provider_config.base_url, provider_config.key, model_config.rpm)
    
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
        
        # 准备额外的提示词，包含对话上下文
        other_prompts = []
        if conversation_context:
            other_prompts.append(f"\n### Previous Conversation ###\n{conversation_context}\n### End of Previous Conversation ###\n")
        
        # 运行引擎 - 传递多模态内容
        engine = UltraThinkEngine(
            client=client,
            model=model_config.model,
            problem_statement=problem_statement_raw,  # 传递多模态内容
            other_prompts=other_prompts,
            max_iterations=model_config.max_iterations,
            required_successful_verifications=model_config.required_verifications,
            num_agents=model_config.num_agent,
            parallel_run_agent=model_config.parallel_run_agent,
            model_stages=model_config.models,
            on_progress=on_progress,
            on_agent_update=on_agent_update,
            llm_params=llm_params,
        )
        
        # 在后台运行引擎
        import asyncio
        engine_task = asyncio.create_task(engine.run())
        
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
    
    else:  # deepthink
        # DeepThink 模式
        progress_queue = []
        
        def on_progress(event):
            """捕获进度事件"""
            progress_queue.append(event)
        
        # 准备额外的提示词，包含对话上下文
        other_prompts = []
        if conversation_context:
            other_prompts.append(f"\n### Previous Conversation ###\n{conversation_context}\n### End of Previous Conversation ###\n")
        
        # 运行引擎 - 传递多模态内容
        engine = DeepThinkEngine(
            client=client,
            model=model_config.model,
            problem_statement=problem_statement_raw,  # 传递多模态内容
            other_prompts=other_prompts,
            max_iterations=model_config.max_iterations,
            required_successful_verifications=model_config.required_verifications,
            model_stages=model_config.models,
            on_progress=on_progress,
            enable_planning=model_config.has_plan_mode,
            llm_params=llm_params,
        )
        
        # 在后台运行引擎
        import asyncio
        engine_task = asyncio.create_task(engine.run())
        
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
        
        # 提取最后一个用户消息作为当前问题
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")
        
        last_user_message = user_messages[-1]
        # 保留原始的多模态内容（如果有图片）
        problem_statement_raw = last_user_message.content
        # 同时提取纯文本版本用于日志和摘要
        problem_statement_text = extract_text_from_content(problem_statement_raw)
        
        # 构建完整的对话上下文（排除最后一条用户消息）
        conversation_context = ""
        if len(request.messages) > 1:
            context_messages = request.messages[:-1]  # 排除最后一条消息
            conversation_parts = []
            for msg in context_messages:
                # 提取文本用于上下文说明
                content_text = extract_text_from_content(msg.content)
                conversation_parts.append(f"{msg.role.upper()}: {content_text}")
            conversation_context = "\n\n".join(conversation_parts)
        
        # 创建后端客户端 - 传入RPM限制，将在每次调用后端API时进行限流
        client = create_client(provider_config.base_url, provider_config.key, model_config.rpm)
        
        # 不再直接注入提示词，而是通过标志传递给引擎
        # 引擎会在正确的时机执行 Ask 和 Plan 阶段
        
        # 准备额外的提示词，包含对话上下文
        other_prompts = []
        if conversation_context:
            other_prompts.append(f"\n### Previous Conversation ###\n{conversation_context}\n### End of Previous Conversation ###\n")
        
        # 根据模型级别处理
        reasoning_text = None
        if model_config.level == "ultrathink":
            engine = UltraThinkEngine(
                client=client,
                model=model_config.model,
                problem_statement=problem_statement_raw,  # 传递多模态内容
                other_prompts=other_prompts,
                max_iterations=model_config.max_iterations,
                required_successful_verifications=model_config.required_verifications,
                num_agents=model_config.num_agent,
                parallel_run_agent=model_config.parallel_run_agent,
                model_stages=model_config.models,
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
                other_prompts=other_prompts,
                max_iterations=model_config.max_iterations,
                required_successful_verifications=model_config.required_verifications,
                model_stages=model_config.models,
                enable_planning=model_config.has_plan_mode,
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

