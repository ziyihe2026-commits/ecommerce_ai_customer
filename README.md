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

## 项目结构

```
ecommerce_ai_customer/
├── main.py             # 主程序（Agent 定义 + 工具 + 交互入口）
├── requirements.txt    # Python 依赖
├── .env                # 环境变量（需自行创建，见下方说明）
├── order.db            # SQLite 数据库（自动生成）
└── chroma_faq/         # ChromaDB 持久化目录（自动生成）
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

## 面试准备要点

如果你把这个项目写进简历，面试官大概率会问：

- **"为什么用 LangGraph 而不用 LangChain 的 AgentExecutor？"**
  → LangGraph 内置 `MemorySaver` 支持多轮对话持久化，且 `create_react_agent` 比老版 `AgentExecutor` 更可控

- **"MemorySaver 的原理是什么？多用户怎么隔离？"**
  → 基于 `thread_id` 做会话隔离，每个 thread 独立存储消息历史；生产环境可换成 `SqliteSaver` 持久化到数据库

- **"情绪识别为什么用关键词而不用模型分类？"**
  → 关键词匹配零延迟、零成本、可解释性强；模型分类需要额外标注数据和推理开销，对客服场景的关键词匹配已足够

- **"FAQ 检索用的什么嵌入模型？为什么选它？"**
  → `text2vec-base-chinese`，中文语义效果好（CoSENT 训练），免费开源，400MB 轻量可本地运行，不需要调 OpenAI API

- **"recursion_limit 设了多少？为什么？"**
  → 10 轮。客服对话通常 2-4 轮工具调用完成，10 轮是安全上限，防止 LLM 陷入"调工具→不满意→再调→还不满意"的死循环
