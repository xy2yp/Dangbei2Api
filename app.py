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
    raise ValueError("未在 .env 文件中找到 API_KEY")

ENABLE_CORS = os.getenv("ENABLE_CORS", "True").lower() in ("true", "1", "yes")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
MAX_CHARS = int(os.getenv("MAX_CHARS", "80000"))  # 从 .env 获取 MAX_CHARS，默认 80000

# 设置日志（单行输出，中文）
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
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
    logger.info("跨域支持已启用")
else:
    logger.info("跨域支持已禁用")

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
            logger.error(f"创建会话失败：HTTP {response.status_code}")
            raise HTTPException(status_code=500, detail="创建会话失败")
        data = response.json()
        if data.get("success"):
            conversation_id = data["data"]["conversationId"]
            logger.info(f"创建新会话：{conversation_id}")
            return conversation_id
        else:
            logger.error(f"创建会话失败：{data}")
            raise HTTPException(status_code=500, detail="创建会话失败")

# 定义授权校验依赖
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

async def check_authorization(authorization: str = Depends(api_key_header)):
    if not authorization:
        logger.error("缺少 Authorization 头部")
        raise HTTPException(status_code=401, detail="缺少 Authorization 头部")
    api_key = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else authorization
    if api_key != API_KEY:
        logger.error(f"无效的 API 密钥：{api_key}")
        raise HTTPException(status_code=401, detail="无效的 API 密钥")
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
        content = re.sub(r"<think>.*?</think>", "", msg.content, flags=re.DOTALL).strip()
        if content:
            concatenated.append(f"{msg.role.capitalize()}: {content}")
    return "\n".join(concatenated)

# 处理 card 类型的内容（优化为表格展示）
def parse_card_content(content: str) -> str:
    try:
        card_data = json.loads(content)
        if card_data.get("cardType") == "DB-CARD-2":
            card_info = card_data.get("cardInfo", {})
            references = []
            for item in card_info.get("cardItems", []):
                if item.get("type") == "2002":  # 搜索来源
                    sources = json.loads(item.get("content", "[]"))
                    for source in sources:
                        id_index = source.get("idIndex", "")
                        name = source.get("name", "")
                        url = source.get("url", "")
                        site_name = source.get("siteName", "")  # 提取 siteName
                        # 格式化为表格行
                        row = f"| {id_index} | [{name}]({url}) | {site_name} |"
                        references.append(row)
            if references:
                # 表格头部
                header = "\n\n| 序号 | 网站URL | 来源 |\n| ---- | ---- | ---- |"
                # 拼接表格
                table = header + "\n" + "\n".join(references)
                return table
            else:
                return "无法解析的新闻内容"
        return "不支持的 card 类型"
    except json.JSONDecodeError:
        logger.warning(f"无法解析 card 内容：{content}")
        return "无法解析的新闻内容"

# 按字符数截断 messages，保留上下文连贯性
def truncate_messages(messages: List[Message], max_chars: int = MAX_CHARS) -> List[Message]:
    # 计算当前总字符数
    total_chars = sum(len(msg.content) for msg in messages)
    if total_chars <= max_chars:
        return messages

    # 保留所有非 user 和 assistant 的消息
    other_messages = [msg for msg in messages if msg.role not in ["user", "assistant"]]
    other_chars = sum(len(msg.content) for msg in other_messages)

    # 可用于 user 和 assistant 的字符数
    available_chars = max_chars - other_chars
    if available_chars <= 0:
        logger.warning("非 user/assistant 消息已超过字符限制，仅保留这些消息")
        return other_messages

    # 获取 user 和 assistant 消息
    ua_messages = [msg for msg in messages if msg.role in ["user", "assistant"]]
    truncated_ua = []
    current_chars = 0

    # 从最新的消息向前累加，直到超出字符限制
    for msg in reversed(ua_messages):
        msg_chars = len(msg.content)
        if current_chars + msg_chars <= available_chars:
            truncated_ua.insert(0, msg)  # 从头插入，保持顺序
            current_chars += msg_chars
        else:
            break

    # 合并结果
    truncated_messages = other_messages + truncated_ua
    logger.info(f"截断上下文：原始字符数 {total_chars}，截断后字符数 {sum(len(msg.content) for msg in truncated_messages)}，消息数 {len(truncated_messages)}")
    return truncated_messages

# 流式响应函数
async def stream_response(request: ChatCompletionRequest, device_id: str, conversation_id: str):
    # 截断 messages
    truncated_messages = truncate_messages(request.messages)
    concatenated_message = concatenate_messages(truncated_messages)
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
    logger.info(f"开始流式响应，会话ID: {conversation_id}，请求: {json.dumps(request.dict(), ensure_ascii=False)}")
    yield generate_chunk(id, created, request.model, {"role": "assistant"})
    thinking = False
    content_parts = []
    card_content = None  # 缓存 card 内容
    is_r1_model = request.model in ["deepseek-r1", "deepseek-r1-search"]

    async with httpx.AsyncClient(http2=True) as client:
        async with client.stream(
            "POST",
            api,
            json=payload,
            headers=headers,
            timeout=1200,
        ) as response:
            if response.status_code != 200:
                error_msg = f"错误：无法获取响应，状态码: {response.status_code}"
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
                            elif data.get("content_type") == "card":
                                parsed_content = parse_card_content(content)
                                if is_r1_model:
                                    card_content = parsed_content  # 缓存 card 内容
                                else:
                                    content_parts.append(parsed_content + "\n\n")
                                    yield generate_chunk(id, created, request.model, {"content": parsed_content + "\n\n"})
                    except json.JSONDecodeError:
                        logger.warning(f"无法解析 JSON 数据：{json_str}")
                        continue
            if thinking:
                content_parts.append("</think>")
                yield generate_chunk(id, created, request.model, {"content": "</think>"})
            if is_r1_model and card_content:
                content_parts.append(card_content + "\n\n")
                yield generate_chunk(id, created, request.model, {"content": card_content + "\n\n"})
            yield generate_chunk(id, created, request.model, {}, "stop")
            content = "".join(content_parts)
            logger.info(f"流式响应完成，会话ID: {conversation_id}，内容: {json.dumps(content, ensure_ascii=False)}")

# 主端点
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, _: None = Depends(check_authorization)):
    logger.info(f"接收到请求: {json.dumps(request.dict(), ensure_ascii=False)}")

    # 校验并设置默认模型
    if request.model not in supported_models:
        request.model = "deepseek-r1"

    # 生成 Device ID
    device_id = generate_device_id()

    # 创建新会话
    conversation_id = await create_conversation(device_id)

    # 截断 messages
    truncated_messages = truncate_messages(request.messages)

    # 处理流式或非流式响应
    if request.stream:
        return StreamingResponse(stream_response(request, device_id, conversation_id), media_type="text/event-stream")
    else:
        concatenated_message = concatenate_messages(truncated_messages)
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
        card_content = None
        thinking = False
        is_r1_model = request.model in ["deepseek-r1", "deepseek-r1-search"]

        async with httpx.AsyncClient(http2=True) as client:
            async with client.stream(
                "POST",
                api,
                json=payload,
                headers=headers,
                timeout=1200,
            ) as response:
                if response.status_code != 200:
                    error_msg = f"无法从 API 获取响应，状态码: {response.status_code}"
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
                                elif data.get("content_type") == "card":
                                    parsed_content = parse_card_content(content)
                                    if is_r1_model:
                                        card_content = parsed_content  # 缓存 card 内容
                                    else:
                                        content_parts.append(parsed_content + "\n\n")
                        except json.JSONDecodeError:
                            logger.warning(f"无法解析 JSON 数据：{json_str}")
                            continue
        if thinking:
            content_parts.append("</think>")
        if is_r1_model and card_content:
            content_parts.append(card_content + "\n\n")
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
        logger.info(f"响应: {json.dumps(response_data, ensure_ascii=False)}")
        return response_data

# /models 端点
@app.get("/v1/models")
async def list_models(_: None = Depends(check_authorization)):
    logger.info("接收到 /models 请求")
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
    logger.info(f"模型响应: {json.dumps(response_data, ensure_ascii=False)}")
    return response_data

# 启动服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)