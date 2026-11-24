# 电商智能客服「小智」—— 基于 LangGraph React Agent

## 项目简介

基于 **LangGraph React Agent** 构建的电商智能客服系统，支持多轮对话记忆、订单查询、FAQ 语义检索、情绪识别与安抚、自动转接人工客服。同时提供 **FastAPI + 网页聊天界面**，可在浏览器中直接使用。

## 技术栈

| 层次 | 技术 |
|------|------|
| Agent 框架 | LangGraph (`create_react_agent`) |
| 多轮记忆 | `MemorySaver`（按会话 ID 隔离上下文） |
| LLM | DeepSeek-Flash (`langchain-deepseek`) |
| 工具调用 | 4 个 `@tool`：订单查询 / FAQ 检索 / 知识库检索 / 情绪检测 |
| 向量检索 | ChromaDB + `text2vec-base-chinese`（免费本地嵌入） |
| 数据库 | SQLite（订单表） |
| 情绪识别 | 关键词匹配（负面词 + 紧急词双重判定） |
| Web 服务 | FastAPI + Uvicorn + 原生 HTML/JS |

## 核心功能

| 功能 | 实现方式 |
|------|---------|
| **多轮对话记忆** | LangGraph `MemorySaver`，按 `thread_id` 隔离会话 |
| **订单查询** | SQLite 本地数据库，`@tool` 封装，LLM 自动按需调用 |
| **FAQ 语义检索** | ChromaDB 向量库 + HuggingFace 免费中文嵌入模型 |
| **情绪识别与安抚** | 关键词匹配（负面词 + 紧急词），Agent 按等级执行安抚话术 |
| **人工客服转接** | 系统提示词中定义 4 条转人工规则，Agent 自动判断 |
| **最大迭代限制** | `recursion_limit=10`，防止 Agent 陷入死循环 |
| **增强版客服封装** | `EnhancedCustomerServiceAgent` 类：情绪预检 → 安抚 → Agent → 转人工判断 → 排队 |
| **Web 聊天界面** | FastAPI 接口 + 单页网页，浏览器直接对话 |
| **知识库上传（RAG）** | 上传 txt/md/pdf/docx → 自动切分（500 字/重叠 50）→ 向量化入库 → Agent 通过 `search_kb` 工具检索 |
| **回归评测** | `eval.py` 内置 50 条场景用例（6 类），输出通过率 / 工具命中率 / 转人工准确率 / P95 时延 |

## 项目结构

```
ecommerce_ai_customer/
├── main.py             # 核心：工具、Agent、EnhancedCustomerServiceAgent
├── kb.py               # 知识库：解析、切分、向量化、search_kb 工具
├── embeddings.py       # 嵌入模型单例（FAQ 与知识库共用）
├── app.py              # FastAPI 入口（/api/chat、/api/kb/*）
├── eval.py             # 回归评测：50 条用例、6 类场景、指标汇总
├── eval_assets/        # 评测用知识库文档（跑完自动从向量库删除）
├── static/
│   └── index.html      # 网页聊天界面
├── requirements.txt    # Python 依赖（已锁定版本）
├── .env                # 环境变量（需自行创建：DEEPSEEK_API_KEY=sk-xxx）
├── .gitignore          # Git 忽略规则
├── order.db            # SQLite 订单数据库（首次运行自动生成）
├── chroma_faq/         # ChromaDB 向量库持久化目录（首次运行自动生成）
└── README.md
```

## 快速开始

### 1. 创建 .env 文件

```bash
# .env
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
```

### 2. 安装依赖

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

> 首次运行会自动下载 `text2vec-base-chinese` 嵌入模型（约 400MB），请耐心等待。

### 3A. 命令行运行

```bash
python main.py
```

程序会自动执行 4 轮演示对话，然后进入手动对话模式。

### 3B. 网页运行（推荐）

```bash
uvicorn app:app --port 8000
```

浏览器打开 <http://127.0.0.1:8000> 即可聊天。首次启动需加载嵌入模型，约 10–30 秒。

### 3C. 使用知识库

页面上方「上传知识库文档」支持 `.txt` / `.md` / `.pdf` / `.docx`，上传后自动切分（500 字/重叠 50）并向量化入库，Agent 会在需要时调用 `search_kb` 检索；文档列表可删除。

| 接口 | 说明 |
|------|------|
| `POST /api/kb/upload` | multipart 上传，字段名 `file` |
| `GET /api/kb/docs` | 列出已入库文档 |
| `DELETE /api/kb/docs/{doc_id}` | 删除文档及其片段 |

## 评测

`eval.py` 内置 50 条场景用例，覆盖 6 类：查订单 10、FAQ 咨询 12、情绪 8、转人工 8、知识库 8、边界 4。判定方式是**必要条件**（该调的工具必须调到、关键词必须出现），属于回归指标，不等于人工评审的"准确率"。

```bash
python eval.py                  # 全量 50 条
python eval.py --limit 3        # 冒烟
python eval.py --cat 情绪        # 只跑某一类
```

最近一次运行（DeepSeek-Flash，本地单机）：

| 指标 | 结果 |
|------|------|
| 整体通过率 | 45/50（90.0%） |
| 工具调用命中率 | 31/32（96.9%） |
| 转人工判定准确率 | 36/40（90.0%） |
| 平均时延 / P95 | 3.26s / 5.32s |

首轮为 43/50（86.0%），定位到"转人工话术整段覆盖模型回复"的缺陷并修复后提升到 90.0%。剩余失败中 4 条属于"模型**提议**转人工被关键词当成**决定**"，1 条是工具路由错选——根治方向见下表。

## 已知局限与改进路线

| 问题 | 现状 | 改进方向 |
|------|------|---------|
| 转人工判定 | 靠匹配回复中的关键词（字符串匹配）；已修复"排队话术覆盖答案"，但仍分不清「提议转人工」和「决定转人工」 | 改为显式 `transfer_to_human` 工具或 LangGraph 条件边，由状态决定控制流 |
| 情绪识别 | 关键词 + 阈值，无法理解反讽/上下文 | 换小模型分类或 LLM 结构化输出 |
| 会话存储 | `MemorySaver` 存在内存，重启即丢 | 换 SQLite / Redis checkpointer |
| 知识库 | 已支持文档上传 + 切分 + 幂等入库；单路向量 Top-3，无重排、无权限控制 | 加混合检索 / rerank、鉴权、按文档过滤 |
| 评测判定 | 关键词 + 工具命中的必要条件，不判语义正确性 | 加 LLM-as-a-Judge 或人工抽检 |
| 工程化 | 有回归评测脚本，但无单元测试、无日志、无鉴权 | 补 pytest、结构化日志、接口鉴权 |
| 部署 | 仅本地运行 | 容器化后部署到云平台 |

## 增强版客服 Agent 的工作流程

```
用户消息
  │
  ▼
① 情绪预检（detect_emotion）
  ├─ 负面情绪 → 安抚话术（拼进返回内容）
  └─ 正常情绪 → 直接进入下一步
  │
  ▼
② 基础 Agent 处理（LangGraph + 工具调用）
  │
  ▼
③ 转人工判断（_need_human_agent）
  ├─ 本会话首次触发 → 排队并返回排队号
  └─ 已转过人工 → 不重复排队
  │
  ▼
④ 异常兜底（try/except）
  └─ Agent 崩溃 → 自动转人工
```
