"""
配置管理模块
负责加载和管理 YAML 配置文件
"""
import os
from typing import Dict, Any, Optional, List
import yaml
from pathlib import Path


class ModelConfig:
    """单个模型的配置"""
    
    def __init__(self, model_id: str, config: Dict[str, Any]):
        self.model_id = model_id
        self.name = config.get("name", model_id)
        self.provider = config.get("provider")
        self.model = config.get("model")
        self.level = config.get("level", "deepthink")  # deepthink, ultrathink
        self.rpm = config.get("rpm")  # 每分钟请求限制
        self.max_iterations = config.get("max_iterations", 30)
        self.required_verifications = config.get("required_verifications", 3)
        self.max_errors = config.get("max_errors_before_give_up", 10)
        self.parallel_check = config.get("parallel_check", False)  # 并行验证模式
        self.max_retry = config.get("max_retry")  # 最大重试次数(可选,不设置则使用系统默认值)
        
        # UltraThink 配置
        self.num_agent = config.get("num_agent")
        self.parallel_run_agent = config.get("parallel_run_agent", 3)
        
        # 特性配置
        self.feature = config.get("feature", {})
        
        # 分阶段模型配置
        self.models = config.get("models", {})
    
    @property
    def has_vision(self) -> bool:
        """是否支持视觉"""
        return self.feature.get("vision", False)
    
    @property
    def has_summary_think(self) -> bool:
        """是否生成思维链摘要"""
        return self.feature.get("summary_think", False)
    
    @property
    def has_plan_mode(self) -> bool:
        """是否启用计划模式"""
        return self.feature.get("plan_mode", False)
    
    @property
    def has_web_search(self) -> bool:
        """是否启用网页搜索"""
        return self.feature.get("web_search", False)
    
    def get_stage_model(self, stage: str) -> str:
        """获取特定阶段的模型,如果未配置则返回主模型"""
        return self.models.get(stage, self.model)
    
    def get_max_retry(self, default: int = 3) -> int:
        """获取最大重试次数,如果未配置则使用提供的默认值"""
        return self.max_retry if self.max_retry is not None else default


class ProviderConfig:
    """提供商配置"""
    
    def __init__(self, provider_id: str, config: Dict[str, Any]):
        self.provider_id = provider_id
        self.base_url = config.get("base_url", "")
        self.key = config.get("key", "")
        self.response_api = config.get("response_api", True)


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
            self._providers[provider_id] = ProviderConfig(provider_id, provider_config)
        
        # 加载模型配置
        models = self._config.get("model", {})
        for model_id, model_config in models.items():
            self._models[model_id] = ModelConfig(model_id, model_config)
    
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

