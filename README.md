# 电商智能客服「小智」—— 基于 LangGraph React Agent

## 项目简介

基于 **LangGraph React Agent** 构建的电商智能客服系统，支持多轮对话记忆、订单查询、FAQ 语义检索、情绪识别与安抚、自动转接人工客服。同时提供 **FastAPI + 网页聊天界面**，可在浏览器中直接使用。

## 技术栈

| 层次 | 技术 |
|------|------|
| Agent 框架 | LangGraph (`create_react_agent`) |
| 多轮记忆 | `MemorySaver`（按会话 ID 隔离上下文） |
| LLM | DeepSeek-Flash (`langchain-deepseek`) |
| 工具调用 | 3 个 `@tool`：订单查询 / FAQ 检索 / 情绪检测 |
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

## 项目结构

```
ecommerce_ai_customer/
├── main.py             # 核心：工具、Agent、EnhancedCustomerServiceAgent
├── app.py              # FastAPI 入口（/api/chat、/api/health）
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

## 已知局限与改进路线

| 问题 | 现状 | 改进方向 |
|------|------|---------|
| 转人工判定 | 靠匹配回复中的关键词（字符串匹配） | 改为 LangGraph 条件边，由状态决定控制流 |
| 情绪识别 | 关键词 + 阈值，无法理解反讽/上下文 | 换小模型分类或 LLM 结构化输出 |
| 会话存储 | `MemorySaver` 存在内存，重启即丢 | 换 SQLite / Redis checkpointer |
| 知识库 | 8 条硬编码 FAQ，无上传入口 | 增加文档上传 + 自动切分 + 幂等入库 |
| 工程化 | 无测试、无日志、无鉴权 | 补 pytest、结构化日志、接口鉴权 |
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
