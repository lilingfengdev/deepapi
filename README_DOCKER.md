# Docker部署指南

## 快速开始

### 1. 准备配置文件
```bash
# 复制示例配置文件（如果有）
cp config.yaml.example config.yaml
# 编辑配置文件
vim config.yaml
```

### 2. 使用Docker运行
```bash
# 运行（使用GitHub镜像）
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### 3. 使用Docker直接运行
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

## 环境变量
- `PYTHONUNBUFFERED=1`: 启用Python日志输出

## 健康检查
容器包含健康检查，可通过以下命令查看状态：
```bash
docker ps --format "table {{.Names}}\t{{.Status}}"
```

## 生产环境建议
1. 使用最新的稳定镜像标签，不要使用`latest`
2. 设置适当的资源限制
3. 配置日志收集
4. 使用HTTPS和适当的认证