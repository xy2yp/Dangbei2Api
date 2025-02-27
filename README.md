# 当贝DeepSeek转API脚本

这是一个将当贝网页deepseek(https://ai.dangbei.com) 转为兼容OpenAI API的python脚本。

基于linux.do论坛 **@云胡不喜** 佬的脚本魔改。

原贴点击[传送门](https://linux.do/t/topic/444507) 。

感谢linux.do论坛 **@yxmiler** 佬提供的保持上下文的思路。

原贴点击[传送门](https://linux.do/t/topic/457926/15?u=jiongjiong_jojo) 。

## 功能说明

**1.支持模型**

- deepseek-r1 模型：`deepseek-r1`
- deepseek-r1 模型，支持搜索功能：`deepseek-r1-search`
- deepseek-v3 模型：`deepseek-v3`
- deepseek-v3 模型，支持搜索功能：`deepseek-v3-search`

## 部署说明

**1.本地部署前修改 .env 文件，配置环境变量**

```plaintext
# API密钥配置（替换为自己的密钥）
API_KEY=sk-your-api-key

# 上下文最大字符数（可选，默认80000，我个人测下来，最高的一次98988字符，没有空响应）
MAX_CHARS=99999

# 是否启用跨域
ENABLE_CORS=True

# 日志级别（可选：DEBUG/INFO/WARNING/ERROR/CRITICAL）
LOG_LEVEL=DEBUG
```

**2.支持Docker部署，可直接使用 Docker 命令**

```bash
docker build -t dangbei2api:latest .
docker run -d -p 8000:8000 -e API_KEY=sk-DangBei666 -e LOG_LEVEL=INFO --name dangbei2api xy2yp/dangbei2api:latest
```


