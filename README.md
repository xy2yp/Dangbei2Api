# 当贝DeepSeek转API脚本

这是一个将当贝网页deepseek(https://ai.dangbei.com) 转为兼容OpenAI API的python脚本。

基于linux.do论坛 **@云胡不喜** 佬的脚本魔改。
原贴点击[传送门](https://linux.do/t/topic/444507) 。

感谢linux.do论坛 **@yxmiler** 佬提供的保持上下文的思路。
原贴点击[传送门](https://linux.do/t/topic/457926/15?u=jiongjiong_jojo) 。

## 支持模型

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
docker run -d -p 8000:8000 -e API_KEY=sk-DangBei666 -e MAX_CHARS=99999  -eENABLE_CORS=True -e LOG_LEVEL=INFO --name dangbei2api xy2yp/dangbei2api:latest
```

# 更新历史

<details>
<summary> ver.20250228</summary>

---

- **优化显示效果**
  - 修改新闻等`card内容`展示方式为表格
- **优化上下文管理**
  - 支持`主动截断`上下文
  - 仅处理user和assistant的内容，保留其他内容，可通过变量配置

---

</details>

---

<details>
<summary> ver.20250227</summary>

---

- **优化联网搜索**
  - deepseek-r1 模型：`deepseek-r1`
  - deepseek-r1 模型，支持搜索功能：`deepseek-r1-search`
  - deepseek-v3 模型：`deepseek-v3`
  - deepseek-v3 模型，支持搜索功能：`deepseek-v3-search`
- **新增功能**
  - 添加`CORS配置`开关
- **功能调整**
  - 移除主动发送`清除上下文`内容的功能
  - 更新`requirements`文件
- **功能调整**
  - 修复`上下文关联失败`问题
  - 修复`签名验证失败`问题
  - 修复`新闻或带 URL 的内容无法解析`问题
  - 将日志长内容换行输出`修改为单行输出`

---

</details>
