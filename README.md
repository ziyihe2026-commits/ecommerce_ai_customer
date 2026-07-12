# 电商智能客服「小智」—— 基于 LangGraph React Agent

## 项目简介

基于 **LangGraph React Agent** 构建的电商智能客服系统，支持多轮对话记忆、订单查询、FAQ 语义检索、情绪识别与安抚、自动转接人工客服。

## 技术栈

| 层次 | 技术 |
|------|------|
| Agent 框架 | LangGraph (`create_react_agent`) |
| 多轮记忆 | `MemorySaver`（按会话 ID 隔离上下文） |
| LLM | DeepSeek-Chat (`langchain-deepseek`) |
| 工具调用 | 3 个 `@tool`：订单查询 / FAQ 检索 / 情绪检测 |
| 向量检索 | ChromaDB + `text2vec-base-chinese`（免费本地嵌入） |
| 数据库 | SQLite（订单表） |
| 情绪识别 | 关键词匹配（负面词 + 紧急词双重判定） |

## 核心功能（全部可运行）

| 功能 | 实现方式 |
|------|---------|
| **多轮对话记忆** | LangGraph `MemorySaver`，按 `thread_id` 隔离会话 |
| **订单查询** | SQLite 本地数据库，`@tool` 封装，LLM 自动按需调用 |
| **FAQ 语义检索** | ChromaDB 向量库 + HuggingFace 免费中文嵌入模型 |
| **情绪识别与安抚** | 关键词匹配（负面词 + 紧急词），Agent 按等级执行安抚话术 |
| **人工客服转接** | 系统提示词中定义 4 条转人工规则，Agent 自动判断 |
| **最大迭代限制** | `recursion_limit=10`，防止 Agent 陷入死循环 |
| **🆕 增强版客服封装** | `EnhancedCustomerServiceAgent` 类：情绪预检 → 安抚 → Agent → 转人工判断 → 排队 |

## 🆕 EnhancedCustomerServiceAgent

增强版客服 Agent 封装，在基础 Agent 之上增加四层防护：

```
用户消息
  │
  ▼
① 情绪预检（detect_emotion）
  ├─ 负面情绪 → 立刻安抚话术
  └─ 正常情绪 → 直接进入下一步
  │
  ▼
② 基础 Agent 处理（LangGraph + 工具调用）
  │
  ▼
③ 转人工判断（_need_human_agent）
  ├─ 匹配转人工关键词 → 走排队流程
  └─ 正常回复 → 直接返回
  │
  ▼
④ 异常兜底（try/except）
  └─ Agent 崩溃 → 自动转人工，避免用户等待超时
```

| 方法 | 说明 |
|------|------|
| `handle_message(user_input, thread_id)` | 增强版消息处理入口 |
| `_need_human_agent(agent_response)` | 检测是否需要转人工 |
| `_transfer_to_human(user_input, thread_id)` | 加入排队队列，返回排队序号 |
| `get_queue_status()` | 查看当前排队人数和预计等待时间 |

## 项目结构

```
ecommerce_ai_customer/
├── main.py             # 主程序（Agent + 工具 + EnhancedCustomerServiceAgent + 交互入口）
├── requirements.txt    # Python 依赖
├── .env                # 环境变量（需自行创建：DEEPSEEK_API_KEY=sk-xxx）
├── .gitignore          # Git 忽略规则
├── order.db            # SQLite 订单数据库（首次运行自动生成）
├── chroma_faq/         # ChromaDB 向量库持久化目录（首次运行自动生成）
└── README.md           # 项目文档
```

## 快速开始

### 1. 创建 .env 文件

```bash
# .env
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

> 首次运行会自动下载 `text2vec-base-chinese` 嵌入模型（约 400MB），请耐心等待。

### 3. 运行

```bash
python main.py
```

程序会自动执行 3 轮演示对话，然后进入手动对话模式。


