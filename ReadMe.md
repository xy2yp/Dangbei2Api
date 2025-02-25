# 当贝DeepSeek转API脚本

这是一个将当贝网页deepseek(https://ai.dangbei.com) 转为兼容OpenAI API的python脚本。

基于linux.do论坛 **@云胡不喜** 佬的脚本魔改。

原贴点击[传送门](https://linux.do/t/topic/444507).

## 功能说明

**1.支持联网搜索**
触发方式：输入结尾添加 **@online** 或者 **@联网** 即可。

**2.支持手动清除上下文**
上下文历史记录默认30，可以自行在环境变量中修改。
手动清除直接发送 **清除上下文** 即可。

## 部署说明

**1.本地部署前修改 .env 文件，配置环境变量**

```plaintext
API_KEY=sk-your_api_key  #随便写一个
MAX_HISTORY=30  #最大对话历史记录数（默认30条）
LOG_LEVEL=DEBUG  日志级别：DEBUG/INFO/WARNING/ERROR/CRITICAL
```

**2.支持Docker部署，可直接使用 Docker 命令**

```bash
docker run -d -p 8000:8000 -e API_KEY=sk-DangBei666 -e MAX_HISTORY=30 -e LOG_LEVEL=INFO --name dangbei2api xy2yp/dangbei2api:latest
```

## 已知问题
思考过程没有被<think>包裹，下午有空摸鱼的时候看看能不能再魔改一下。

```

```


