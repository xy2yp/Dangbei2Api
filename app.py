# 导入所需库
import os  # 读取环境变量
import secrets  # 生成安全随机数
import time     # 时间处理
import uuid     # 生成唯一ID
import hashlib  # 哈希加密
import json     # JSON数据处理
import httpx    # 异步HTTP请求
import logging  # 日志记录
import random  # 随机数生成
from typing import AsyncGenerator, List, Dict, Union  # 类型提示
from pydantic import BaseModel, Field  # 数据验证模型
from fastapi import FastAPI, HTTPException, Header  # Web框架组件
from fastapi.responses import StreamingResponse  # 流式响应支持
from collections import OrderedDict  # 有序字典
from datetime import datetime  # 日期时间处理


# 配置日志记录（INFO级别）
#logging.basicConfig(level=logging.INFO)
#logger = logging.getLogger(__name__)

# 从环境变量获取日志级别（默认DEBUG）
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()

# 配置日志记录
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# 创建FastAPI应用实例
app = FastAPI()

# 配置类：存储应用程序配置参数
#class Config(BaseModel):
#     API密钥配置（默认值）
#    API_KEY: str = Field(
#        default="skgUXNcLwm0rnnEt55Mg6yp68",
#        description="用于身份验证的API密钥"
#    )

class Config(BaseModel):
    API_KEY: str = Field(
        default=os.getenv("API_KEY", "sk_gUXNcLwm0rnnEt55Mg8hq88"),
        description="用于身份验证的API密钥"
    )
    
    # 最大历史记录数（默认保留10条对话历史）
#    MAX_HISTORY: int = Field(
#        default=10,
#        description="保留的最大对话历史记录数"
#    )
    
    MAX_HISTORY: int = Field(
        default=int(os.getenv("MAX_HISTORY", 30)),  # 支持环境变量配置
        description="保留的最大对话历史记录数"
    )

    # API请求域名配置
    API_DOMAIN: str = Field(
        default="https://ai-api.dangbei.net",
        description="API请求的基础域名"
    )
    
    # 用户代理字符串（模拟浏览器请求）
    USER_AGENT: str = Field(
        default="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
        description="HTTP请求的User-Agent头"
    )

# 创建全局配置实例
config = Config()

# 辅助函数：验证API密钥
async def verify_api_key(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="缺少API密钥")
    
    # 提取Bearer Token格式的API密钥
    api_key = authorization.replace("Bearer ", "").strip()
    if api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="无效的API密钥")
    return api_key

# 消息模型：定义对话消息的结构
class Message(BaseModel):
    role: str    # 消息角色（如user/assistant）
    content: str  # 消息内容

# 聊天请求模型：定义聊天接口的请求参数
class ChatRequest(BaseModel):
    model: str           # 使用的模型名称
    messages: List[Message]  # 消息历史列表
    stream: bool = False  # 是否启用流式响应

# 对话历史管理类
class ChatHistory:
    def __init__(self):
        self.current_device_id = None      # 当前设备ID
        self.current_conversation_id = None  # 当前对话ID
        self.conversation_count = 0        # 当前设备对话计数

    # 获取或创建设备ID和对话ID
    def get_or_create_ids(self, force_new=False) -> tuple[str, str]:
        # 当需要强制新建或达到对话上限时，生成新设备ID
        if force_new or not self.current_device_id or self.conversation_count >= config.MAX_HISTORY:
            self.current_device_id = self._generate_device_id()
            self.current_conversation_id = None
            self.conversation_count = 0

        return self.current_device_id, self.current_conversation_id

    # 添加新对话记录
    def add_conversation(self, conversation_id: str):
        if not self.current_device_id:
            return
        self.current_conversation_id = conversation_id
        self.conversation_count += 1

    # 生成符合特定格式的设备ID（UUID + NanoID组合）
    def _generate_device_id(self) -> str:
        uuid_str = uuid.uuid4().hex
        nanoid_str = ''.join(random.choices(
            "useandom26T198340PX75pxJACKVERYMINDBUSHWOLF_GQZbfghjklqvwyzrict",
            k=20
        ))
        return f"{uuid_str}_{nanoid_str}"

# 核心处理管道类
class Pipe:
    def __init__(self):
        self.data_prefix = "data:"  # SSE数据前缀
        self.user_agent = config.USER_AGENT  # 用户代理
        self.chat_history = ChatHistory()  # 对话历史管理实例

    # 主处理流程（异步生成器）
    async def pipe(self, body: dict) -> AsyncGenerator[Dict, None]:
        thinking_state = {"thinking": -1}  # 思考状态跟踪
        user_message = body["messages"][-1]["content"]  # 获取最新用户消息

        # 检查是否需要清除上下文
        force_new_context = user_message == "清除上下文"

        # 获取设备ID和对话ID
        device_id, conversation_id = self.chat_history.get_or_create_ids(force_new_context)

        # 如果需要新建对话
        if not conversation_id:
            conversation_id = await self._create_conversation(device_id)
            if not conversation_id:
                yield {"error": "无法创建新对话"}
                return
            self.chat_history.add_conversation(conversation_id)

        # 处理清除上下文指令
        if force_new_context:
            yield {"choices": [{"delta": {"content": "上下文已清除，开始新的对话。"}, "finish_reason": None}]}
            yield {"choices": [{"delta": {"meta": {
                "device_id": device_id,
                "conversation_id": conversation_id
            }}, "finish_reason": None}]}
            return

        # 确定用户操作类型（是否需要联网）
        user_action = "deep"
        if user_message.rstrip().endswith(("@online", "@联网")):
            user_action = "deep,online"
            user_message = user_message.rstrip().rsplit("@", 1)[0].rstrip()

        # 构建请求负载
        payload = {
            "stream": True,
            "botCode": "AI_SEARCH",
            "userAction": user_action,
            "model": "deepseek",
            "conversationId": conversation_id,
            "question": user_message,
        }

        # 生成请求签名所需参数
        timestamp = str(int(time.time()))
        nonce = self._nanoid(21)
        sign = self._generate_sign(timestamp, payload, nonce)

        # 构建请求头
        headers = {
            "Origin": "https://ai.dangbei.com",
            "Referer": "https://ai.dangbei.com/",
            "User-Agent": self.user_agent,
            "deviceId": device_id,
            "nonce": nonce,
            "sign": sign,
            "timestamp": timestamp,
        }

        # 构建完整API地址
        api = f"{config.API_DOMAIN}/ai-search/chatApi/v1/chat"

        try:
            # 发起异步HTTP流式请求
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", api, json=payload, headers=headers, timeout=1200) as response:
                    if response.status_code != 200:
                        error = await response.aread()
                        yield {"error": self._format_error(response.status_code, error)}
                        return

                    # 处理流式响应
                    async for line in response.aiter_lines():
                        if not line.startswith(self.data_prefix):
                            continue

                        json_str = line[len(self.data_prefix):]
                        try:
                            data = json.loads(json_str)
                        except json.JSONDecodeError as e:
                            yield {"error": f"JSON解析错误: {str(e)}", "data": json_str}
                            return

                        # 处理回答类型数据
                        if data.get("type") == "answer":
                            content = data.get("content")
                            content_type = data.get("content_type")
                            
                            # 处理思考状态转换
                            if thinking_state["thinking"] == -1 and content_type == "thinking":
                                thinking_state["thinking"] = 0
                                yield {"choices": [{"delta": {"content": " \n"}, "finish_reason": None}]}
                            elif thinking_state["thinking"] == 0 and content_type == "text":
                                thinking_state["thinking"] = 1
                                yield {"choices": [{"delta": {"content": "\n \n\n"}, "finish_reason": None}]}
                            
                            if content:
                                yield {"choices": [{"delta": {"content": content}, "finish_reason": None}]}

                    # 返回元数据
                    yield {"choices": [{"delta": {"meta": {
                        "device_id": device_id,
                        "conversation_id": conversation_id
                    }}, "finish_reason": None}]}

        except Exception as e:
            logger.error(f"管道处理错误: {str(e)}")
            yield {"error": self._format_exception(e)}

    # 格式化HTTP错误信息
    def _format_error(self, status_code: int, error: bytes) -> str:
        error_str = error.decode(errors="ignore") if isinstance(error, bytes) else error
        return json.dumps({"error": f"HTTP {status_code}: {error_str}"}, ensure_ascii=False)

    # 格式化异常信息
    def _format_exception(self, e: Exception) -> str:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"}, ensure_ascii=False)

    # 生成NanoID（自定义字母表）
    def _nanoid(self, size=21) -> str:
        url_alphabet = "useandom-26T198340PX75pxJACKVERYMINDBUSHWOLF_GQZbfghjklqvwyzrict"
        random_bytes = secrets.token_bytes(size)
        return "".join([url_alphabet[b & 63] for b in reversed(random_bytes)])

    # 生成请求签名（MD5加密）
    def _generate_sign(self, timestamp: str, payload: dict, nonce: str) -> str:
        payload_str = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        sign_str = f"{timestamp}{payload_str}{nonce}"
        return hashlib.md5(sign_str.encode("utf-8")).hexdigest().upper()

    # 创建新对话会话
    async def _create_conversation(self, device_id: str) -> str:
        payload = {"botCode": "AI_SEARCH"}
        timestamp = str(int(time.time()))
        nonce = self._nanoid(21)
        sign = self._generate_sign(timestamp, payload, nonce)

        headers = {
            "Origin": "https://ai.dangbei.com",
            "Referer": "https://ai.dangbei.com/",
            "User-Agent": self.user_agent,
            "deviceId": device_id,
            "nonce": nonce,
            "sign": sign,
            "timestamp": timestamp,
        }

        api = f"{config.API_DOMAIN}/ai-search/conversationApi/v1/create"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(api, json=payload, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        return data["data"]["conversationId"]
        except Exception as e:
            logger.error(f"创建对话失败: {str(e)}")
        return None

# 创建管道实例
pipe = Pipe()

# 聊天接口端点（兼容OpenAI API）
@app.post("/v1/chat/completions")
async def chat(request: ChatRequest, authorization: str = Header(None)):
    await verify_api_key(authorization)

    # 流式响应生成器
    async def response_generator():
        thinking_content = []
        is_thinking = False
        
        async for chunk in pipe.pipe(request.model_dump()):
            if "choices" in chunk and chunk["choices"]:
                delta = chunk["choices"][0]["delta"]
                if "content" in delta:
                    content = delta["content"]
                    # 处理思考状态标记
                    if content == " \n":
                        is_thinking = True
                    elif content == "\n \n\n":
                        is_thinking = False
                    # 收集思考内容
                    if is_thinking and content != " \n":
                        thinking_content.append(content)

                        
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    # 根据请求类型返回不同响应
    if request.stream:
        return StreamingResponse(response_generator(), media_type="text/event-stream")

    # 非流式响应处理
    content = ""
    meta = None
    try:
        async for chunk in pipe.pipe(request.model_dump()):
            if "choices" in chunk and chunk["choices"]:
                delta = chunk["choices"][0]["delta"]
                if "content" in delta:
                    content += delta["content"]
                if "meta" in delta:
                    meta = delta["meta"]
    except Exception as e:
        logger.error(f"聊天请求处理错误: {str(e)}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

    # 处理内容格式
    parts = content.split("\n\n\n", 1)
    reasoning_content = parts[0] if len(parts) > 0 else ""
    content = parts[1] if len(parts) > 1 else ""

    # 格式化推理内容
    if reasoning_content:
        start_idx = reasoning_content.find(" ")
        end_idx = reasoning_content.rfind(" ")
        
        if start_idx != -1 and end_idx != -1:
            inner_content = reasoning_content[start_idx + 7:end_idx].strip()
            inner_content = inner_content.replace(" ", "").replace(" ", "").strip()
            reasoning_content = f" \n{inner_content}\n "
        else:
            reasoning_content = reasoning_content.replace(" ", "").replace(" ", "").strip()
            reasoning_content = f" \n{reasoning_content}\n "

    return {
        "id": str(uuid.uuid4()),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "message": {
                "role": "assistant",
                "reasoning_content": reasoning_content,
                "content": content,
                "meta": meta
            },
            "finish_reason": "stop"
        }]
    }

# 模型信息接口
@app.get("/v1/models")
async def get_models(authorization: str = Header(None)):
    await verify_api_key(authorization)
    
    current_time = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": "deepseek",
                "object": "model",
                "created": current_time,
                "owned_by": "library"
            }
        ]
    }

# 启动服务（开发模式）
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
