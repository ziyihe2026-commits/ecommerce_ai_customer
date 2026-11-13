"""FastAPI 入口：把 main.py 的客服 Agent 暴露成 HTTP 服务 + 网页聊天界面"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from main import EnhancedCustomerServiceAgent, init_db, init_faq

agent = EnhancedCustomerServiceAgent()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    init_faq()  # 首次需加载嵌入模型，约 30 秒
    yield


app = FastAPI(title="电商智能客服 小智", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "web"


class ChatResponse(BaseModel):
    reply: str


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    reply = agent.handle_message(req.message, thread_id=req.thread_id)
    return ChatResponse(reply=reply)


# 必须最后挂载：让 /api/* 优先匹配，其余交给静态文件
app.mount("/", StaticFiles(directory="static", html=True), name="static")
