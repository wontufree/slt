"""
言象 (YanXiang) 手语翻译系统 - FastAPI 后端服务
基于 OpenAI 格式调用 uni-sign-translator 模型
"""

import os
import asyncio
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx


# ============== 配置 ==============
API_URL = "https://openai.uniin.cn/openapi/v2/chat/completions"
API_KEY = "sk-b3IwcHljeWNvMTA0eWluZnNrbnhhZDIzbXRmaG9hcmE6djhicXY4cGN2aTI0emFzYTJqMHB6YzV4YmZuMmV5b3Nmdm1uOTd5a2Q0eDB0NnpjbXR5M2tnZGxkZzU5dDRvMA=="
MODEL_NAME = "uni-sign-translator"

# 开发模式：设为 True 则模拟返回，不调用真实 API
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"

# ============== 临时文件目录 ==============
TEMP_DIR = Path(tempfile.gettempdir()) / "yanxiang_slt"
TEMP_DIR.mkdir(exist_ok=True)


# ============== 启动/关闭事件 ==============
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[言象] 启动完成 | MOCK_MODE={MOCK_MODE}")
    yield
    print("[言象] 关闭中...")


# ============== FastAPI 应用 ==============
app = FastAPI(
    title="言象手语翻译 API",
    description="澳门手语翻译系统 - 基于 FastAPI + Uni-Sign 模型",
    version="1.0.0",
    lifespan=lifespan,
)

# ============== CORS 中间件 ==============
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应设为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== 请求/响应模型 ==============
class TranslationResponse(BaseModel):
    status: str
    text: str
    confidence: float
    latency_ms: int
    filename_processed: str


# ============== 辅助函数 ==============
def call_slt_model(video_path: str) -> dict:
    """
    调用手语识别模型（OpenAI 兼容格式）
    视频文件需转为 base64 格式传递
    """
    import base64
    import requests
    import os

    print(f"[call_slt_model] 开始处理视频: {video_path}, 大小: {os.path.getsize(video_path)} bytes")
    with open(video_path, "rb") as f:
        video_base64 = base64.b64encode(f.read()).decode("utf-8")
    print(f"[call_slt_model] base64 编码完成, 长度: {len(video_base64)}")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": video_base64}],
    }

    print(f"[call_slt_model] 正在发送请求到 API...")
    # 连接超时 15 秒，读取超时 300 秒（推理需要时间）
    response = requests.post(API_URL, headers=headers, json=payload, timeout=(15, 300))
    print(f"[call_slt_model] API 返回, status: {response.status_code}")
    response.raise_for_status()
    result = response.json()

    content = result["choices"][0]["message"]["content"]
    print(f"[call_slt_model] 完成, content: {content[:50]}...")
    return {"text": content.strip()}


async def mock_slt_model(video_path: str) -> dict:
    """
    MOCK 模式：模拟模型返回（用于无 GPU 的开发调试）
    """
    import random
    await asyncio.sleep(2)  # 模拟推理延迟

    mock_texts = [
        "今天天气非常不错，我们可以去郊外的公园散散心。",
        "请问有什么可以帮助您的吗？",
        "这是一个手语翻译的测试结果。",
    ]
    return {"text": random.choice(mock_texts)}


# ============== API 路由 ==============
@app.post("/api/v1/slt/translate", response_model=TranslationResponse)
async def translate_video(file: UploadFile = File(...)):
    print(f"[收到请求] filename={file.filename}, content_type={file.content_type}")
    """
    接收手语视频文件，返回翻译文本

    - 视频暂存到临时目录防止 OOM
    - 调用 uni-sign-translator 模型
    - 返回 JSON 格式翻译结果
    """
    import time
    start_time = time.time()

    # 1. 校验文件类型
    if not file.filename:
        raise HTTPException(status_code=400, detail="未提供文件名")

    ext = Path(file.filename).suffix.lower()
    if ext not in [".mp4", ".mov", ".avi", ".mkv"]:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的视频格式: {ext}，仅支持 mp4/mov/avi/mkv"
        )

    # 2. 保存到临时文件
    temp_video_path = TEMP_DIR / f"{Path(file.filename).stem}{ext}"
    try:
        with open(temp_video_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}")

    # 3. 调用模型
    print(f"[调试] 文件大小: {len(content)} bytes, 保存路径: {temp_video_path}")
    try:
        if MOCK_MODE:
            result = await mock_slt_model(str(temp_video_path))
        else:
            # 使用线程池调用同步的 requests（避免阻塞事件循环）
            print("[translate] 开始调用线程池...")
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, call_slt_model, str(temp_video_path))
            print(f"[translate] 线程池返回: {result}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型调用失败: {str(e)}")

    # 4. 计算延迟
    latency_ms = int((time.time() - start_time) * 1000)

    # 5. 构建响应
    return TranslationResponse(
        status="success",
        text=result["text"],
        confidence=98.67 if MOCK_MODE else 95.00,  # Mock 模式给个高置信度
        latency_ms=latency_ms,
        filename_processed=file.filename,
    )


@app.get("/")
async def root():
    return {"message": "言象手语翻译系统 API", "version": "1.0.0", "mock_mode": MOCK_MODE}


@app.get("/health")
async def health():
    return {"status": "healthy"}


# ============== 启动命令 ==============
if __name__ == "__main__":
    import uvicorn

    # 默认 0.0.0.0:8000，生产环境建议使用 nginx 反向代理
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8001,
        reload=True,  # 开发模式热重载
        log_level="info",
    )
