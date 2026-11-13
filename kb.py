"""知识库：文档解析 → 切分 → 向量化 → 语义检索（RAG）"""

import hashlib
from pathlib import Path

from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embeddings import get_embeddings

DOCS_DIR = Path("kb_docs")
PERSIST_DIR = "./chroma_kb"
SUPPORTED = {".txt", ".md", ".markdown", ".pdf", ".docx"}

_kb = None


def get_kb() -> Chroma:
    global _kb
    if _kb is None:
        _kb = Chroma(
            collection_name="kb_docs",  # Chroma 要求集合名 3-512 字符
            embedding_function=get_embeddings(),
            persist_directory=PERSIST_DIR,
        )
    return _kb


def _read_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if suffix == ".docx":
        from docx import Document

        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError(f"不支持的文件类型：{suffix}")


def ingest_file(path: Path) -> dict:
    """解析 → 切分 → 入库。按内容哈希生成稳定 ID，重复上传自动替换旧片段。"""
    text = _read_text(path).strip()
    if not text:
        raise ValueError("文件里没有可抽取的文字（可能是扫描版 PDF）")

    doc_id = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
    metadatas = [
        {"doc_id": doc_id, "source": path.name, "chunk": i}
        for i in range(len(chunks))
    ]

    kb = get_kb()
    kb.delete(where={"doc_id": doc_id})  # 幂等：先删旧片段再写入
    kb.add_texts(texts=chunks, metadatas=metadatas, ids=ids)

    return {
        "doc_id": doc_id,
        "source": path.name,
        "chars": len(text),
        "chunks": len(chunks),
    }


def list_docs() -> list[dict]:
    data = get_kb().get(include=["metadatas"])
    docs: dict[str, dict] = {}
    for md in data.get("metadatas") or []:
        if not md:
            continue
        item = docs.setdefault(
            md["doc_id"], {"doc_id": md["doc_id"], "source": md["source"], "chunks": 0}
        )
        item["chunks"] += 1
    return sorted(docs.values(), key=lambda d: d["source"])


def delete_doc(doc_id: str) -> None:
    get_kb().delete(where={"doc_id": doc_id})


@tool
def search_kb(question: str) -> str:
    """在客服知识库中检索相关文档片段（售后政策、发票、保修、活动规则、商品说明等）。
    适用场景：用户询问 FAQ 之外的政策类、说明类问题。
    Args:
        question: 用户问题的原文
    """
    kb = get_kb()
    if not kb.get(limit=1)["ids"]:
        return "知识库暂未上传任何文档。"
    docs = kb.similarity_search(question, k=3)
    if not docs:
        return "知识库中没有找到相关内容，建议转接人工客服。"
    return "\n\n".join(
        f"[来源：{d.metadata.get('source', '未知')}]\n{d.page_content}" for d in docs
    )
