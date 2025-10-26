"""
Deep Think API 主应用
FastAPI 应用程序入口
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from config import config
from api.v1 import chat, models

# 配置日志
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("Starting Deep Think API...")
    logger.info(f"Loaded {len(config.list_models())} models")
    yield
    logger.info("Shutting down Deep Think API...")


# 创建 FastAPI 应用
app = FastAPI(
    title="Deep Think API",
    description="OpenAI-compatible API with DeepThink and UltraThink reasoning engines",
    version="1.0.0",
    lifespan=lifespan,
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat.router, tags=["Chat"])
app.include_router(models.router, tags=["Models"])


@app.get("/")
async def root():
    """根端点"""
    return {
        "name": "Deep Think API",
        "version": "1.0.0",
        "description": "OpenAI-compatible API with DeepThink and UltraThink reasoning engines",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "models": "/v1/models",
        }
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
    )

