"""FastAPI 入口：把 main.py 的客服 Agent 暴露成 HTTP 服务 + 网页聊天界面"""

import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from kb import DOCS_DIR, SUPPORTED, delete_doc, ingest_file, list_docs
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


@app.post("/api/kb/upload")
async def kb_upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(400, f"不支持的文件类型：{suffix}，仅支持 {', '.join(sorted(SUPPORTED))}")

    DOCS_DIR.mkdir(exist_ok=True)
    target = DOCS_DIR / file.filename
    with target.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        return ingest_file(target)
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(400, f"解析失败：{exc}") from exc


@app.get("/api/kb/docs")
def kb_list():
    return list_docs()


@app.delete("/api/kb/docs/{doc_id}")
def kb_delete(doc_id: str):
    delete_doc(doc_id)
    return {"deleted": doc_id}


# 必须最后挂载：让 /api/* 优先匹配，其余交给静态文件
app.mount("/", StaticFiles(directory="static", html=True), name="static")
