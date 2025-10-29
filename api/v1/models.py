"""
模型列表 API
OpenAI 兼容的 /v1/models 端点
"""
from fastapi import APIRouter, Header, HTTPException
from typing import List, Dict, Any

from config import config
from .chat import verify_auth  # 复用，别重复造轮子

router = APIRouter()


@router.get("/v1/models")
async def list_models(authorization: str = Header(None)):
    """
    列出所有可用模型
    OpenAI 兼容的端点
    """
    verify_auth(authorization)
    
    models = config.list_models()
    
    return {
        "object": "list",
        "data": models
    }


@router.get("/v1/models/{model_id}")
async def get_model(model_id: str, authorization: str = Header(None)):
    """
    获取特定模型信息
    OpenAI 兼容的端点
    """
    verify_auth(authorization)
    
    model_config = config.get_model(model_id)
    
    if not model_config:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    return {
        "id": model_id,
        "object": "model",
        "created": 1677610602,
        "owned_by": model_config.provider,
        "permission": [],
        "root": model_id,
        "parent": None,
    }

