import uuid
import time
import json
import hashlib
import secrets
import re
import logging
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.security import APIKeyHeader
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Literal, Optional
import httpx
from dotenv import load_dotenv
import os

# 加载 .env 文件
load_dotenv()

# 获取环境变量
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY not found in .env file")

ENABLE_CORS = os.getenv("ENABLE_CORS", "True").lower() in ("true", "1", "yes")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# 设置日志
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),  # 动态设置日志等级
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 初始化 FastAPI 应用
app = FastAPI()

# 根据配置启用跨域
if ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("CORS enabled")
else:
    logger.info("CORS disabled")

# 定义常量
api_domain = "https://ai-api.dangbei.net"
user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"

# 支持的模型和对应的 userAction 映射
supported_models = ["deepseek-r1", "deepseek-r1-search", "deepseek-v3", "deepseek-v3-search"]
model_to_user_action = {
    "deepseek-r1": ["deep"],
    "deepseek-r1-search": ["deep", "online"],
    "deepseek-v3": [],
    "deepseek-v3-search": ["online"],
}


# 工具函数
def nanoid(size=21):
    url_alphabet = "useandom-26T198340PX75pxJACKVERYMINDBUSHWOLF_GQZbfghjklqvwyzrict"
    return "".join(secrets.choice(url_alphabet) for _ in range(size))


def generate_device_id():
    return f"{uuid.uuid4().hex}_{nanoid(20)}"


def generate_sign(timestamp: str, payload: dict, nonce: str) -> str:
    payload_str = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    sign_str = f"{timestamp}{payload_str}{nonce}"
    return hashlib.md5(sign_str.encode("utf-8")).hexdigest().upper()


# 创建会话
async def create_conversation(device_id: str) -> str:
    payload = {"botCode": "AI_SEARCH"}
    timestamp = str(int(time.time()))
    nonce = nanoid(21)
    sign = generate_sign(timestamp, payload, nonce)
    headers = {
        "Origin": "https://ai.dangbei.com",
        "Referer": "https://ai.dangbei.com/",
        "User-Agent": user_agent,
        "deviceId": device_id,
        "nonce": nonce,
        "sign": sign,
        "timestamp": timestamp,
        "content-type": "application/json",
    }
    api = f"{api_domain}/ai-search/conversationApi/v1/create"
    async with httpx.AsyncClient(http2=True) as client:
        response = await client.post(api, json=payload, headers=headers)
        if response.status_code != 200:
            logger.error(f"Failed to create conversation: HTTP {response.status_code}")
            raise HTTPException(status_code=500, detail="Failed to create conversation")
        data = response.json()
        if data.get("success"):
            conversation_id = data["data"]["conversationId"]
            logger.info(f"Created new conversation: {conversation_id}")
            return conversation_id
        else:
            logger.error(f"Failed to create conversation: {data}")
            raise HTTPException(status_code=500, detail="Failed to create conversation")


# 定义授权校验依赖
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


async def check_authorization(authorization: str = Depends(api_key_header)):
    if not authorization:
        logger.error("Authorization header missing")
        raise HTTPException(status_code=401, detail="Authorization header missing")
    api_key = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else authorization
    if api_key != API_KEY:
        logger.error(f"Invalid API key: {api_key}")
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


# 定义请求模型
class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    frequency_penalty: Optional[float] = 0
    presence_penalty: Optional[float] = 0
    temperature: Optional[float] = 1
    top_p: Optional[float] = 1


# 生成流式响应块
def generate_chunk(id: str, created: int, model: str, delta: dict, finish_reason: Optional[str] = None):
    chunk = {
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason
            }
        ]
    }
    return f"data: {json.dumps(chunk)}\n\n"


# 拼接 messages 数组为字符串
def concatenate_messages(messages: List[Message]) -> str:
    concatenated = []
    for msg in messages:
        # 移除 <think> 标签及其内容
        content = re.sub(r"<think>.*?</think>", "", msg.content, flags=re.DOTALL).strip()
        if content:  # 仅添加非空内容
            concatenated.append(f"{msg.role.capitalize()}: {content}")
    return "\n".join(concatenated)


# 流式响应函数
async def stream_response(request: ChatCompletionRequest, device_id: str, conversation_id: str):
    concatenated_message = concatenate_messages(request.messages)
    user_action = model_to_user_action[request.model]
    payload = {
        "stream": True,
        "botCode": "AI_SEARCH",
        "userAction": ",".join(user_action),
        "model": "deepseek",
        "conversationId": conversation_id,
        "question": concatenated_message,
    }
    timestamp = str(int(time.time()))
    nonce = nanoid(21)
    sign = generate_sign(timestamp, payload, nonce)
    headers = {
        "Origin": "https://ai.dangbei.com",
        "Referer": "https://ai.dangbei.com/",
        "User-Agent": user_agent,
        "deviceId": device_id,
        "nonce": nonce,
        "sign": sign,
        "timestamp": timestamp,
    }
    api = f"{api_domain}/ai-search/chatApi/v1/chat"
    id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    logger.info(
        f"Streaming response started for conversation: {conversation_id}, request: {json.dumps(request.dict())}")
    yield generate_chunk(id, created, request.model, {"role": "assistant"})
    thinking = False
    content_parts = []
    async with httpx.AsyncClient(http2=True) as client:
        async with client.stream(
            "POST",
            api,
            json=payload,
            headers=headers,
            timeout=1200,
        ) as response:
            if response.status_code != 200:
                error_msg = f"Error: Failed to get response, status: {response.status_code}"
                logger.error(error_msg)
                yield generate_chunk(id, created, request.model, {"content": error_msg}, "stop")
                return
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    json_str = line[5:]
                    try:
                        data = json.loads(json_str)
                        content = data.get("content") or ""
                        if content:
                            content = re.sub(r"<details>.*?</details>", "", content, flags=re.DOTALL)
                            if data.get("content_type") == "thinking":
                                if not thinking:
                                    thinking = True
                                    content_parts.append("<think>")
                                    yield generate_chunk(id, created, request.model, {"content": "<think>"})
                                content_parts.append(content)
                                yield generate_chunk(id, created, request.model, {"content": content})
                            elif data.get("content_type") == "text":
                                if thinking:
                                    thinking = False
                                    content_parts.append("</think>")
                                    yield generate_chunk(id, created, request.model, {"content": "</think>"})
                                content_parts.append(content)
                                yield generate_chunk(id, created, request.model, {"content": content})
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON: {json_str}")
                        continue
            if thinking:
                content_parts.append("</think>")
                yield generate_chunk(id, created, request.model, {"content": "</think>"})
            yield generate_chunk(id, created, request.model, {}, "stop")
            content = "".join(content_parts)
            logger.info(f"Streaming response completed for conversation: {conversation_id}, content: {content}")


# 主端点
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, _: None = Depends(check_authorization)):
    logger.info(f"Received request: {json.dumps(request.dict())}")

    # 校验并设置默认模型
    if request.model not in supported_models:
        request.model = "deepseek-r1"

    # 生成 Device ID
    device_id = generate_device_id()

    # 创建新会话
    conversation_id = await create_conversation(device_id)

    # 处理流式或非流式响应
    if request.stream:
        return StreamingResponse(stream_response(request, device_id, conversation_id), media_type="text/event-stream")
    else:
        concatenated_message = concatenate_messages(request.messages)
        user_action = model_to_user_action[request.model]
        payload = {
            "stream": True,
            "botCode": "AI_SEARCH",
            "userAction": ",".join(user_action),
            "model": "deepseek",
            "conversationId": conversation_id,
            "question": concatenated_message,
        }
        timestamp = str(int(time.time()))
        nonce = nanoid(21)
        sign = generate_sign(timestamp, payload, nonce)
        headers = {
            "Origin": "https://ai.dangbei.com",
            "Referer": "https://ai.dangbei.com/",
            "User-Agent": user_agent,
            "deviceId": device_id,
            "nonce": nonce,
            "sign": sign,
            "timestamp": timestamp,
        }
        api = f"{api_domain}/ai-search/chatApi/v1/chat"
        content_parts = []
        thinking = False
        async with httpx.AsyncClient(http2=True) as client:
            async with client.stream(
                "POST",
                api,
                json=payload,
                headers=headers,
                timeout=1200,
            ) as response:
                if response.status_code != 200:
                    error_msg = f"Failed to get response from API, status: {response.status_code}"
                    logger.error(error_msg)
                    raise HTTPException(status_code=500, detail=error_msg)
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        json_str = line[5:]
                        try:
                            data = json.loads(json_str)
                            content = data.get("content") or ""
                            if content:
                                content = re.sub(r"<details>.*?</details>", "", content, flags=re.DOTALL)
                                if data.get("content_type") == "thinking":
                                    if not thinking:
                                        thinking = True
                                        content_parts.append("<think>")
                                    content_parts.append(content)
                                elif data.get("content_type") == "text":
                                    if thinking:
                                        thinking = False
                                        content_parts.append("</think>")
                                    content_parts.append(content)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse JSON: {json_str}")
                            continue
        if thinking:
            content_parts.append("</think>")
        content = "".join(content_parts)

        response_data = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        logger.info(f"Response: {json.dumps(response_data)}")
        return response_data


# /models 端点
@app.get("/v1/models")
async def list_models(_: None = Depends(check_authorization)):
    logger.info("Received request for /models")
    models = [
        {
            "id": model,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "dangbei"
        }
        for model in supported_models
    ]
    response_data = {
        "object": "list",
        "data": models
    }
    logger.info(f"Models response: {json.dumps(response_data)}")
    return response_data


# 启动服务
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
