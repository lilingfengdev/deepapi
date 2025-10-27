# Deep Think API

Python 版本的 Deep Think 推理引擎 API，提供 OpenAI 兼容的接口，支持 DeepThink 和 UltraThink 两种深度推理模式。

## 特性

* ✨ **OpenAI 兼容 API** - 完全兼容 OpenAI Chat Completion API
* 🧠 **DeepThink 模式** - 单 Agent 深度迭代推理，连续验证确保质量
* 🚀 **UltraThink 模式** - 多 Agent 并行探索，综合多角度分析
* ⚡ **RPM 限制** - 灵活的每分钟请求数限制，区分快慢模型
* 💭 **Summary Think** - 流式返回友好的思维链，提升用户体验
* 🎯 **多模型支持** - 支持任何 OpenAI 兼容的 LLM 提供商
* 📊 **分阶段模型** - 不同推理阶段可使用不同模型优化成本

## 快速开始

### 1. 安装依赖

```bash
cd deepapi
pip install -r requirements.txt
```

### 2. 配置

复制配置文件模板：

```bash
cp config.yaml.example config.yaml
```

编辑 `config.yaml` 填写你的配置：

```yaml
system:
  key: "your-api-key-here"  # API 访问密钥
  host: "0.0.0.0"
  port: 8000

provider:
  openai:
    base_url: "https://api.openai.com/v1"
    key: "sk-xxx"

model:
  gpt-4o-deepthink:
    name: "GPT-4O Deep Think"
    provider: openai
    model: gpt-4o
    level: deepthink  # deepthink | ultrathink
    rpm: 10  # 每分钟请求限制
    feature:
      summary_think: true  # 启用思维链展示
```

### 3. 运行服务

#### 本地部署

```bash
python main.py
```

#### Docker 部署

##### 使用 Docker Compose（推荐）

```bash
# 使用GitHub镜像运行
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

##### 使用 Docker 直接运行

```bash
# 使用GitHub镜像直接运行
docker run -d \
  --name deepapi \
  -p 8000:8000 \
  -v $(pwd)/config.yaml:/app/config.yaml:ro \
  ghcr.io/zhongruan0522/deepapi:latest

# 如果需要本地构建
docker build -t deepapi .
docker run -d \
  --name deepapi \
  -p 8000:8000 \
  -v $(pwd)/config.yaml:/app/config.yaml:ro \
  deepapi
```

##### 环境变量说明
- `PYTHONUNBUFFERED=1`: 启用Python日志输出
- 自定义DNS服务器（解决网络问题）：`8.8.8.8`, `8.8.4.4`

##### 健康检查
容器包含健康检查，可通过以下命令查看状态：
```bash
docker ps --format "table {{.Names}}\t{{.Status}}"
```

##### 生产环境建议
1. 使用稳定版本镜像标签（如 `beta` 或具体版本号），避免使用 `latest`
2. 设置适当的资源限制
3. 配置日志收集
4. 使用HTTPS和适当的认证

## API 使用

### 聊天补全

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "gpt-4o-deepthink",
    "messages": [
      {"role": "user", "content": "解释量子纠缠"}
    ],
    "stream": true
  }'
```

### 列出模型

```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer your-api-key"
```

### 使用 OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-api-key"
)

# 流式请求
stream = client.chat.completions.create(
    model="gpt-4o-deepthink",
    messages=[{"role": "user", "content": "解释相对论"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='')
```

## 配置详解

### 模型级别 (level)

- **`deepthink`** - DeepThink 模式
  - 单 Agent 深度推理
  - 多轮迭代验证
  - 连续 3 次验证通过才输出
  
- **`ultrathink`** - UltraThink 模式
  - 多 Agent 并行探索
  - 从不同角度分析问题
  - 最终综合所有 Agent 结果

### RPM 限制

**重要**: RPM 限制用于控制**后端调用 LLM API 的频率**，而非限制用户请求频率。

由于 DeepThink/UltraThink 在一次用户请求中会多次调用后端 LLM，RPM 限制可以防止触发后端 API 的速率限制。

```yaml
model:
  fast-model:
    rpm: 50  # 快速模型,高速率限制（每分钟最多50次后端调用）
  
  slow-model:
    rpm: 10  # 慢速模型,低速率限制（每分钟最多10次后端调用）
  
  unlimited-model:
    # rpm 不设置则不限制后端调用频率
```

### Summary Think 功能

启用 `summary_think` 后，在流式响应开始时会先返回伪造的思维链：

```
<thinking>

Initializing Deep Think Engine...

Problem: 解释量子纠缠...

Round 1 - Initial Analysis
  • Understanding problem structure
  • Identifying key constraints
  • Generating initial approach

Round 2 - Refinement & Verification
  • Reviewing previous reasoning
  • Addressing identified gaps
  • Verifying solution correctness

Preparing final answer...

</thinking>

[实际的答案内容]
```

这提升了用户体验，让用户在等待时能看到"AI 正在思考"的过程。

### 分阶段模型

为不同推理阶段指定不同模型以优化成本：

```yaml
model:
  hybrid-model:
    provider: openai
    model: gpt-4o  # 主模型
    models:
      initial: gpt-4o-mini        # 初始思考用便宜模型
      improvement: gpt-4o         # 改进用强模型
      verification: gpt-4o-mini   # 验证用便宜模型
      correction: gpt-4o          # 修正用强模型
      summary: gpt-4o             # 总结用强模型
```

UltraThink 还支持：

```yaml
models:
  planning: gpt-4o              # 计划生成
  agent_config: gpt-4o          # Agent 配置
  agent_thinking: gpt-4o-mini   # Agent 思考(多个并行,用便宜的)
  synthesis: gpt-4o             # 结果综合
  summary: gpt-4o               # 最终总结
```

## 架构说明

### DeepThink 流程

```
问题输入 → 初始思考 → 自我改进 → 验证
    ↓                              ↓
    └─────── 修正 ←── 验证失败 ←────┘
                       ↓
                 验证通过(3次) → 输出答案
```

### UltraThink 流程

```
问题输入 → 生成计划 → 生成 Agent 配置
                          ↓
    Agent 1: 角度1 ──┐
    Agent 2: 角度2 ──┤ 并行执行 DeepThink
    ...             ├→ 每个都经过验证
    Agent N: 角度N ──┘
            ↓
    综合所有结果 → 输出最佳方案
```

## 架构说明

### DeepThink 流程