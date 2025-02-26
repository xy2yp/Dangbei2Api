# 当贝DeepSeek转API脚本

这是一个将当贝网页deepseek(https://ai.dangbei.com) 转为兼容OpenAI API的python脚本。

基于linux.do论坛 **@云胡不喜** 佬的脚本魔改。

原贴点击[传送门](https://linux.do/t/topic/444507) 。

## 功能说明

**1.支持模型**

- deepseek-r1 模型：`deepseek-r1`
- deepseek-r1 模型，支持搜索功能：`deepseek-r1-search`
- deepseek-v1 模型：`deepseek-v1`
- deepseek-v1 模型，支持搜索功能：`deepseek-v1-search`

## 部署说明

**1.本地部署前修改 .env 文件，配置环境变量**

```plaintext
API_KEY=sk-your_api_key  #随便写一个
MAX_HISTORY=10  #最大对话历史记录数（默认30条）
LOG_LEVEL=INFO  日志级别：DEBUG/INFO/WARNING/ERROR/CRITICAL
```

**2.支持Docker部署，可直接使用 Docker 命令**

```bash
docker build -t dangbei2api:latest .
docker run -d -p 8000:8000 -e API_KEY=sk-DangBei666 -e LOG_LEVEL=INFO --name dangbei2api dangbei2api:latest
```

## 已知问题
