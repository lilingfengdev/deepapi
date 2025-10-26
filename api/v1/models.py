"""
模型列表 API
OpenAI 兼容的 /v1/models 端点
"""
from fastapi import APIRouter, Header, HTTPException
from typing import List, Dict, Any

from config import config

router = APIRouter()


def verify_auth(authorization: str = Header(None)) -> bool:
    """验证 API 密钥"""
    if not config.api_key:
        return True
    
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    
    token = authorization.replace("Bearer ", "").strip()
    
    if not config.validate_api_key(token):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return True


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

