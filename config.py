"""
配置管理模块
负责加载和管理 YAML 配置文件
"""
import os
from typing import Dict, Any, Optional, List
import yaml
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """单个模型的配置"""
    model_id: str
    name: str
    provider: str
    model: str
    level: str = "deepthink"  # deepthink, ultrathink
    rpm: Optional[int] = None  # 每分钟请求限制
    max_iterations: int = 30
    required_verifications: int = 3
    max_errors: int = 10
    parallel_check: bool = False  # 并行验证模式
    max_retry: Optional[int] = None  # 最大重试次数
    
    # UltraThink 配置
    num_agent: Optional[int] = None
    parallel_run_agent: int = 3
    
    # 特性配置 - 直接用布尔值，别搞那些花里胡哨的
    has_vision: bool = False
    has_summary_think: bool = False
    has_plan_mode: bool = False
    has_web_search: bool = False
    
    # 分阶段模型配置
    models: Dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, model_id: str, config: Dict[str, Any]) -> 'ModelConfig':
        """从字典创建配置"""
        feature = config.get("feature", {})
        return cls(
            model_id=model_id,
            name=config.get("name", model_id),
            provider=config.get("provider"),
            model=config.get("model"),
            level=config.get("level", "deepthink"),
            rpm=config.get("rpm"),
            max_iterations=config.get("max_iterations", 30),
            required_verifications=config.get("required_verifications", 3),
            max_errors=config.get("max_errors_before_give_up", 10),
            parallel_check=config.get("parallel_check", False),
            max_retry=config.get("max_retry"),
            num_agent=config.get("num_agent"),
            parallel_run_agent=config.get("parallel_run_agent", 3),
            has_vision=feature.get("vision", False),
            has_summary_think=feature.get("summary_think", False),
            has_plan_mode=feature.get("plan_mode", False),
            has_web_search=feature.get("web_search", False),
            models=config.get("models", {})
        )
    
    def get_stage_model(self, stage: str) -> str:
        """获取特定阶段的模型"""
        return self.models.get(stage, self.model)
    
    def get_max_retry(self, default: int = 3) -> int:
        """获取最大重试次数"""
        return self.max_retry if self.max_retry is not None else default


@dataclass
class ProviderConfig:
    """提供商配置"""
    provider_id: str
    base_url: str = ""
    key: str = ""
    response_api: bool = True
    
    @classmethod
    def from_dict(cls, provider_id: str, config: Dict[str, Any]) -> 'ProviderConfig':
        """从字典创建配置"""
        return cls(
            provider_id=provider_id,
            base_url=config.get("base_url", ""),
            key=config.get("key", ""),
            response_api=config.get("response_api", True)
        )


class Config:
    """全局配置管理器"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.config_path = Path("config.yaml")
            self._config: Dict[str, Any] = {}
            self._models: Dict[str, ModelConfig] = {}
            self._providers: Dict[str, ProviderConfig] = {}
            self.load()
    
    def load(self, config_path: Optional[Path] = None):
        """加载配置文件"""
        if config_path:
            self.config_path = config_path
        
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"配置文件不存在: {self.config_path}\n"
                f"请复制 config.yaml.example 为 config.yaml 并填写配置"
            )
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
        
        # 加载提供商配置
        providers = self._config.get("provider", {})
        for provider_id, provider_config in providers.items():
            self._providers[provider_id] = ProviderConfig.from_dict(provider_id, provider_config)
        
        # 加载模型配置
        models = self._config.get("model", {})
        for model_id, model_config in models.items():
            self._models[model_id] = ModelConfig.from_dict(model_id, model_config)
    
    @property
    def api_key(self) -> str:
        """系统API密钥"""
        return self._config.get("system", {}).get("key", "")
    
    @property
    def host(self) -> str:
        """服务器host"""
        return self._config.get("system", {}).get("host", "0.0.0.0")
    
    @property
    def port(self) -> int:
        """服务器端口"""
        return self._config.get("system", {}).get("port", 8000)
    
    @property
    def log_level(self) -> str:
        """日志级别"""
        return self._config.get("system", {}).get("log_level", "INFO")
    
    @property
    def max_retry(self) -> int:
        """默认最大重试次数"""
        return self._config.get("system", {}).get("max_retry", 3)
    
    def get_model(self, model_id: str) -> Optional[ModelConfig]:
        """获取模型配置"""
        return self._models.get(model_id)
    
    def get_provider(self, provider_id: str) -> Optional[ProviderConfig]:
        """获取提供商配置"""
        return self._providers.get(provider_id)
    
    def list_models(self) -> List[Dict[str, Any]]:
        """列出所有可用模型"""
        return [
            {
                "id": model_id,
                "object": "model",
                "created": 1677610602,
                "owned_by": model.provider,
                "permission": [],
                "root": model_id,
                "parent": None,
            }
            for model_id, model in self._models.items()
        ]
    
    def validate_api_key(self, key: str) -> bool:
        """验证API密钥"""
        if not self.api_key:
            # 如果未设置密钥,则不验证
            return True
        return key == self.api_key


# 全局配置实例
config = Config()

