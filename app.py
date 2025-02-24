import secrets
import time
import uuid
import hashlib
import json
import httpx
import logging
from typing import AsyncGenerator, List, Dict, Union
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
from collections import OrderedDict
from datetime import datetime
import random

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 配置
class Config(BaseModel):
    # API 密钥
    API_KEY: str = Field(
        default="sk_dangbei666",
        description="API key for authentication"
    )
    
    # 最大历史记录数
    MAX_HISTORY: int = Field(
        default=10,
        description="Maximum number of conversation histories to keep"
    )
    
    # API 域名
    API_DOMAIN: str = Field(
        default="https://ai-api.dangbei.net",
        description="API Domain for requests"
    )
    
    # User Agent
    USER_AGENT: str = Field(
        default="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
        description="User agent string for requests"
    )

# 创建全局配置实例
config = Config()

# 辅助函数：验证 API 密钥
async def verify_api_key(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing API key")
    
    api_key = authorization.replace("Bearer ", "").strip()
    if api_key != config.API_KEY:  # 使用配置中的 API_KEY
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = False

class ChatHistory:
    def __init__(self):
        self.history = OrderedDict()
        self.current_device_id = None
        self.conversation_count = 0

    def get_or_create_ids(self) -> tuple[str, str]:
        """获取或创建新的 device_id 和 conversation_id"""
        # 如果没有当前设备ID或对话次数达到上限，创建新的设备ID
        if not self.current_device_id or self.conversation_count >= config.MAX_HISTORY:
            self.current_device_id = self._generate_device_id()
            self.conversation_count = 0
            self.history.clear()  # 清空历史记录

        # 获取当前设备的会话ID
        conversation_id = self.history.get(self.current_device_id, {}).get("conversation_id")
        
        return self.current_device_id, conversation_id

    def add_conversation(self, conversation_id: str):
        """添加新的对话记录"""
        if not self.current_device_id:
            return
            
        self.history[self.current_device_id] = {
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat()
        }
        self.conversation_count += 1

    def _generate_device_id(self) -> str:
        """生成新的设备ID"""
        uuid_str = uuid.uuid4().hex
        nanoid_str = ''.join(random.choices(
            "useandom26T198340PX75pxJACKVERYMINDBUSHWOLF_GQZbfghjklqvwyzrict",
            k=20
        ))
        return f"{uuid_str}_{nanoid_str}"

class Pipe:
    def __init__(self):
        self.data_prefix = "data:"
        self.user_agent = config.USER_AGENT
        self.chat_history = ChatHistory()

    async def pipe(self, body: dict) -> AsyncGenerator[Dict, None]:
        thinking_state = {"thinking": -1}
        
        # 获取或创建设备ID和会话ID
        device_id, conversation_id = self.chat_history.get_or_create_ids()

        # 如果没有会话ID，创建新的会话
        if not conversation_id:
            conversation_id = await self._create_conversation(device_id)
            if not conversation_id:
                yield {"error": "Failed to create conversation"}
                return
            # 保存新的对话记录
            self.chat_history.add_conversation(conversation_id)

        user_message = body["messages"][-1]["content"]
        payload = {
            "stream": True,
            "botCode": "AI_SEARCH",
            "userAction": "deep",
            "model": "deepseek",
            "conversationId": conversation_id,
            "question": user_message,
        }

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

        api = f"{config.API_DOMAIN}/ai-search/chatApi/v1/chat"  # 使用配置中的 API_DOMAIN

        try:
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", api, json=payload, headers=headers, timeout=1200) as response:
                    if response.status_code != 200:
                        error = await response.aread()
                        yield {"error": self._format_error(response.status_code, error)}
                        return

                    async for line in response.aiter_lines():
                        if not line.startswith(self.data_prefix):
                            continue

                        json_str = line[len(self.data_prefix):]

                        try:
                            data = json.loads(json_str)
                        except json.JSONDecodeError as e:
                            yield {"error": f"JSONDecodeError: {str(e)}", "data": json_str}
                            return

                        if data.get("type") == "answer":
                            content = data.get("content")
                            content_type = data.get("content_type")
                            
                            # 处理思考状态
                            if thinking_state["thinking"] == -1 and content_type == "thinking":
                                thinking_state["thinking"] = 0
                                yield {"choices": [{"delta": {"content": "<think>\n"}, "finish_reason": None}]}
                            elif thinking_state["thinking"] == 0 and content_type == "text":
                                thinking_state["thinking"] = 1
                                yield {"choices": [{"delta": {"content": "\n</think>\n\n"}, "finish_reason": None}]}
                            
                            if content:
                                yield {"choices": [{"delta": {"content": content}, "finish_reason": None}]}

                    # 在最后添加元数据
                    yield {"choices": [{"delta": {"meta": {
                        "device_id": device_id,
                        "conversation_id": conversation_id
                    }}, "finish_reason": None}]}

        except Exception as e:
            logger.error(f"Error in pipe: {str(e)}")
            yield {"error": self._format_exception(e)}

    def _format_error(self, status_code: int, error: bytes) -> str:
        error_str = error.decode(errors="ignore") if isinstance(error, bytes) else error
        return json.dumps({"error": f"HTTP {status_code}: {error_str}"}, ensure_ascii=False)

    def _format_exception(self, e: Exception) -> str:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"}, ensure_ascii=False)

    def _nanoid(self, size=21) -> str:
        url_alphabet = "useandom-26T198340PX75pxJACKVERYMINDBUSHWOLF_GQZbfghjklqvwyzrict"
        random_bytes = secrets.token_bytes(size)
        return "".join([url_alphabet[b & 63] for b in reversed(random_bytes)])

    def _generate_sign(self, timestamp: str, payload: dict, nonce: str) -> str:
        payload_str = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        sign_str = f"{timestamp}{payload_str}{nonce}"
        return hashlib.md5(sign_str.encode("utf-8")).hexdigest().upper()

    async def _create_conversation(self, device_id: str) -> str:
        """创建新的会话"""
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
            logger.error(f"Error creating conversation: {str(e)}")
        return None

# 创建实例
pipe = Pipe()

@app.post("/v1/chat/completions")
async def chat(request: ChatRequest, authorization: str = Header(None)):
    """
    OpenAI API 兼容的 Chat 端点
    """
    await verify_api_key(authorization)

    async def response_generator():
        async for chunk in pipe.pipe(request.model_dump()):
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    if request.stream:
        return StreamingResponse(response_generator(), media_type="text/event-stream")

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
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

    parts = content.split("\n\n\n", 1)
    reasoning_content = parts[0] if len(parts) > 0 else ""
    content = parts[1] if len(parts) > 1 else ""

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

@app.get("/v1/models")
async def get_models(authorization: str = Header(None)):
    # 验证 API 密钥
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

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
