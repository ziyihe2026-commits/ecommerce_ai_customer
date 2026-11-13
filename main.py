"""
基于 LangGraph React Agent 的电商智能客服系统

核心功能（全部可运行验证）：
1. 多轮对话记忆 —— MemorySaver 持久化上下文
2. 订单查询工具 —— SQLite 本地订单数据库
3. FAQ 语义检索 —— ChromaDB + 免费本地嵌入模型
4. 情绪识别与安抚 —— 关键词匹配 + 安抚话术
5. 人工客服转接 —— 无法解决时自动引导
6. 最大迭代限制 —— recursion_limit 防止无限循环
"""

from dotenv import load_dotenv
import os
import sqlite3
from datetime import datetime
from typing import Optional

# ── LangChain / LangGraph ──────────────────────────────
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_deepseek import ChatDeepSeek
from langchain_chroma import Chroma
from langchain.tools import tool
from langchain_core.messages import SystemMessage

from embeddings import get_embeddings
from kb import search_kb

# ── 加载环境变量 ────────────────────────────────────────
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# ================================================================
# 1. 数据库初始化 —— 订单表 + 测试数据
# ================================================================
def init_db():
    """创建订单表并插入测试数据"""
    conn = sqlite3.connect("order.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id   TEXT PRIMARY KEY,
            user_name  TEXT,
            goods_name TEXT,
            price      REAL,
            status     TEXT
        )
    """)
    test_data = [
        ("ORD001", "张三", "无线蓝牙耳机", 199.0, "已发货"),
        ("ORD002", "李四", "机械键盘",     129.0, "待付款"),
        ("ORD003", "王五", "电竞鼠标",      89.0, "已完成"),
        ("ORD004", "赵六", "USB-C 充电线",   29.9, "已发货"),
        ("ORD005", "张三", "手机壳",         39.0, "待发货"),
    ]
    cursor.executemany(
        "INSERT OR IGNORE INTO orders VALUES (?,?,?,?,?)", test_data
    )

    # 追加 100 条种子订单（ORD006–ORD105），让演示数据接近真实业务量
    import random

    random.seed(42)
    names = ["张三", "李四", "王五", "赵六", "孙七", "周八", "吴九", "郑十"]
    goods = [
        "无线蓝牙耳机", "机械键盘", "电竞鼠标", "USB-C 充电线", "手机壳",
        "显示器支架", "便携充电宝", "降噪耳塞", "桌面音箱", "Type-C 扩展坞",
    ]
    statuses = ["待付款", "待发货", "已发货", "已完成", "已取消"]
    extra = [
        (
            f"ORD{i:03d}",
            random.choice(names),
            random.choice(goods),
            round(random.uniform(19.0, 599.0), 1),
            random.choice(statuses),
        )
        for i in range(6, 106)
    ]
    cursor.executemany("INSERT OR IGNORE INTO orders VALUES (?,?,?,?,?)", extra)
    conn.commit()
    conn.close()


# ================================================================
# 2. FAQ 知识库 —— ChromaDB 语义检索（免费本地嵌入）
# ================================================================
FAQ_DATA: list[dict] = []
FAQ_COLLECTION: Optional[Chroma] = None

def init_faq():
    """构建 FAQ 向量库，使用免费中文嵌入模型（首次运行自动下载）"""
    global FAQ_DATA, FAQ_COLLECTION

    FAQ_DATA.clear()
    FAQ_DATA.extend([
        {
            "question": "如何查询我的订单？",
            "answer": "请提供您的订单号（如 ORD001），我立刻帮您查询。"
        },
        {
            "question": "怎么申请退换货？",
            "answer": "收到商品 7 天内可申请退换货。请提供订单号，我为您生成退换货申请链接。"
        },
        {
            "question": "退款什么时候到账？",
            "answer": "退款将在仓库签收退货后 3-5 个工作日内原路返回您的支付账户。"
        },
        {
            "question": "发货后多久能收到？",
            "answer": "正常情况下 2-5 个工作日送达，具体时效视收货地址而定。"
        },
        {
            "question": "商品有质量问题怎么办？",
            "answer": "非常抱歉！请提供订单号和商品照片，我立即帮您走售后流程。"
        },
        {
            "question": "可以修改收货地址吗？",
            "answer": "订单未发货前可修改地址，请提供订单号和新地址。"
        },
        {
            "question": "蓝牙耳机怎么连接？",
            "answer": "长按电源键 3 秒进入配对模式，手机蓝牙搜索设备名 BT-XXX 点击连接即可。"
        },
        {
            "question": "客服工作时间是什么？",
            "answer": "在线客服工作时间：每天 9:00-22:00。其余时间可留言，次日回复。"
        },
    ])

    texts = [item["question"] for item in FAQ_DATA]
    # 幂等初始化：已有集合就复用，避免每次启动重复追加（原先 2→4 的问题）
    FAQ_COLLECTION = Chroma(
        collection_name="faq",
        embedding_function=get_embeddings(),
        persist_directory="./chroma_faq",
    )
    if FAQ_COLLECTION.get(limit=1)["ids"]:
        return
    FAQ_COLLECTION.add_texts(texts=texts)


# ================================================================
# 3. 工具定义
# ================================================================

@tool
def query_order(order_id: str) -> str:
    """
    根据订单号查询用户订单信息（含用户、商品、价格、状态）。
    适用场景：用户提供订单号希望查进度/详情。
    Args:
        order_id: 订单编号，如 ORD001
    """
    conn = sqlite3.connect("order.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return (
            f"订单号：{row[0]} | 用户：{row[1]} | "
            f"商品：{row[2]} | 金额：{row[3]}元 | 状态：{row[4]}"
        )
    return f"未查询到订单 {order_id}，请确认订单号是否正确。"


@tool
def search_faq(question: str) -> str:
    """
    从 FAQ 知识库中语义检索与用户问题最匹配的答案。
    适用场景：用户咨询退换货规则、时效、售后政策等通用问题。
    Args:
        question: 用户提出的问题原文
    """
    global FAQ_COLLECTION
    if FAQ_COLLECTION is None:
        return "FAQ 知识库尚未初始化，请稍后再试。"
    docs = FAQ_COLLECTION.similarity_search(question, k=2)
    if not docs:
        return "FAQ 库中暂未找到相关答案，建议转接人工客服处理。"
    results = []
    for i, doc in enumerate(docs, 1):
        matched_q = doc.page_content
        full_answer = next(
            (item["answer"] for item in FAQ_DATA if item["question"] == matched_q),
            "请参考 FAQ 文档。"
        )
        results.append(f"[匹配{i}] 问题：{matched_q}\n     答案：{full_answer}")
    return "\n".join(results)


@tool
def detect_emotion(user_message: str) -> str:
    """
    检测用户消息中的负面情绪（投诉/愤怒/焦虑）。
    适用场景：用户语气激烈时调用，决定是否需要安抚话术。
    Args:
        user_message: 用户最近一条消息原文
    """
    # 注意："退款/退货"是正常业务意图，不是负面情绪词，放进来会导致
    # 用户问"退款多久到账"被判成负面情绪并触发安抚话术。
    negative_kw = [
        "投诉", "差评", "垃圾", "骗子", "坑", "气死", "烦", "烂",
        "坑爹", "假的", "骗钱", "太差", "坑人", "曝光", "维权", "12315",
    ]
    urgent_kw = ["马上", "立刻", "赶紧", "快", "急", "现在就要"]

    neg_hits = sum(1 for kw in negative_kw if kw in user_message)
    urg_hits = sum(1 for kw in urgent_kw if kw in user_message)

    if neg_hits >= 2 and urg_hits >= 1:
        return "情绪等级：高（强烈不满+急迫）。请立即安抚并优先处理，必要时转人工。"
    elif neg_hits >= 1:
        return "情绪等级：中（含有负面词汇）。请先表达歉意再解决问题。"
    elif urg_hits >= 1:
        return "情绪等级：低（用户比较着急）。加快响应节奏。"
    else:
        return "情绪等级：正常。标准服务流程即可。"


# ================================================================
# 4. 系统提示词 —— 定义 Agent 行为规范
# ================================================================
SYSTEM_PROMPT = """
你是一个专业的电商客服 AI，名字叫"小智"。严格遵守以下规则：

【你的工具】
- query_order：查询订单（需要订单号）
- search_faq：语义搜索 FAQ 知识库（退换货/时效/政策等通用问题）
- search_kb：检索已上传的客服文档（发票、保修、活动规则、商品说明等）
- detect_emotion：检测用户情绪

【工作流程——每轮必须执行】
1. 先调用 detect_emotion 检测用户当前消息的情绪：
   - 等级"高" → 先说安抚话术（如"非常抱歉给您带来不好的体验，我马上帮您处理"）
   - 等级"中" → 先说"很抱歉给您添麻烦了，我来帮您解决"
   - 等级"低"或"正常" → 直接解决问题
2. 判断用户意图（查订单 / 问政策 / 投诉）
3. 调用对应工具获取信息
4. 用自然语言完整回答
5. 结尾问"还有其他需要帮您的吗？"

【转人工规则——遇到以下任一情况，必须提示转人工】
- 用户要求退款/赔偿，且 FAQ 中没有明确方案
- 工具返回空结果且你无法通过常识回答
- 用户连续表达不满
- 涉及法律/隐私/账号安全问题
转人工话术："这个我暂时无法直接处理，为您转接人工客服，请稍候。"

【禁止行为】
- 禁止编造订单数据（必须调用工具查询）
- 禁止承诺具体赔偿金额
- 不确定就诚实说不知道，转人工
"""


# ================================================================
# 5. 构建 Agent（带记忆）
# ================================================================
_agent = None

def get_agent():
    """懒加载全局 Agent（含 MemorySaver 和工具集）"""
    global _agent
    if _agent is not None:
        return _agent

    llm = ChatDeepSeek(
        model="deepseek-flash",
        api_key=DEEPSEEK_API_KEY,
        temperature=0.3,
    )

    tools = [query_order, search_faq, detect_emotion, search_kb]

    # MemorySaver —— 按 thread_id 隔离多轮对话上下文
    memory = MemorySaver()

    _agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SystemMessage(content=SYSTEM_PROMPT),
        checkpointer=memory,
    )
    return _agent


# ================================================================
# 6. 对话接口
# ================================================================
def chat(user_input: str, thread_id: str = "default") -> str:
    """
    单轮对话入口。

    Parameters
    ----------
    user_input : 用户输入
    thread_id  : 会话 ID。不同用户用不同 ID 实现记忆隔离；
                同一 thread_id 共享最近多轮上下文。
    """
    agent = get_agent()

    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 10,  # 最大工具调用轮数
    }

    result = agent.invoke(
        {"messages": [("user", user_input)]},
        config=config,
    )

    return result["messages"][-1].content


# ================================================================
# 7. 增强版客服 Agent 封装 —— 情绪优先 + 智能转人工
# ================================================================
class EnhancedCustomerServiceAgent:
    """
    增强版客服 Agent，在基础 Agent 之上增加：

    【新增能力】
    1. 情绪预检 —— 先于 Agent 检测用户情绪，负面情绪立即安抚
    2. 智能转人工 —— Agent 回复中包含转人工意图时自动走排队流程
    3. 人工客服排队 —— 维护转人工队列，支持排队序号查询
    4. 异常兜底 —— Agent 出错时自动转人工，避免用户等待超时
    """

    def __init__(self):
        self.human_agent_queue: list[dict] = []
        # 已转人工的会话：同一会话只排队一次，避免追问时被反复转接
        self.transferred_threads: set[str] = set()

    def handle_message(self, user_input: str, thread_id: str = "default") -> str:
        """
        增强版消息处理入口。

        工作流程：
        ① 情绪预检（detect_emotion）
        ② 负面/紧急情绪 → 安抚话术
        ③ 调用基础 Agent 处理
        ④ 判断是否需要转人工
        ⑤ 返回最终结果

        Parameters
        ----------
        user_input : 用户输入
        thread_id  : 会话 ID，用于多轮对话记忆隔离

        Returns
        -------
        str : 客服回复
        """
        # ── ① 情绪预检 ──
        try:
            emotion_result = detect_emotion.invoke({"user_message": user_input})
        except Exception:
            emotion_result = "情绪等级：正常。标准服务流程即可。"

        # ── ② 负面 / 紧急情绪 → 先安抚（拼进返回值，Web 端才拿得到）──
        soothing = ""
        if "高" in emotion_result:
            soothing = "非常抱歉给您带来这么不好的体验，我马上优先为您处理！\n"
        elif "中" in emotion_result:
            soothing = "很抱歉给您添麻烦了，我来帮您解决。\n"

        # ── ③ Agent 处理 ──
        try:
            response = chat(user_input, thread_id)

            # ── ④ 判断是否需要转人工 ──
            if self._need_human_agent(response):
                if thread_id in self.transferred_threads:
                    # 已转过人工，不再重复排队，直接把模型回复返回
                    return soothing + response
                return soothing + self._transfer_to_human(user_input, thread_id)

            return soothing + response
        except Exception as e:
            # ── ⑤ 异常兜底转人工 ──
            print(f"[系统] Agent 处理异常：{e}")
            return soothing + self._transfer_to_human(user_input, thread_id)

    def _need_human_agent(self, agent_response: str) -> bool:
        """
        检测 Agent 回复中是否包含转人工意图。

        匹配关键词：
        - "转人工" / "转接人工" → Agent 主动建议转接
        - "人工客服" → 涉及需要人工处理的场景
        - "无法处理" / "无法直接处理" → Agent 能力边界
        """
        keywords = ["转人工", "转接人工", "人工客服", "无法处理", "无法直接处理"]
        return any(kw in agent_response for kw in keywords)

    def _transfer_to_human(self, user_input: str, thread_id: str) -> str:
        """
        记录转人工请求并返回排队信息。

        将用户输入、会话 ID、时间戳加入人工客服队列，
        返回排队序号供用户参考。
        """
        self.transferred_threads.add(thread_id)
        self.human_agent_queue.append({
            "user_input": user_input,
            "session_id": thread_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        queue_num = len(self.human_agent_queue)
        return (
            f"我为您转接人工客服，请稍等...\n"
            f"您当前排队序号：{queue_num}，"
            f"人工客服将在工作时间（每天 9:00-22:00）内尽快回复您。"
        )

    def get_queue_status(self) -> str:
        """查看当前排队情况"""
        if not self.human_agent_queue:
            return "当前无人排队。"
        return f"当前排队人数：{len(self.human_agent_queue)}，预计等待约 {len(self.human_agent_queue) * 3} 分钟。"


# ================================================================
# 8. 命令行入口
# ================================================================
if __name__ == "__main__":
    if not DEEPSEEK_API_KEY:
        print("错误：请在 .env 文件中设置 DEEPSEEK_API_KEY=你的key")
        exit(1)

    print("初始化数据库...")
    init_db()

    print("初始化 FAQ 知识库（首次运行会下载嵌入模型约 400MB，请耐心等待）...")
    init_faq()

    print()
    print("=" * 55)
    print("   电商智能客服「小智」已上线")
    print("   支持：查订单 | FAQ 咨询 | 退换货 | 情绪安抚 | 转人工")
    print("   输入 quit / exit 退出")
    print("=" * 55)

    # ── 创建增强版客服 Agent ──
    enhanced_agent = EnhancedCustomerServiceAgent()

    # ── 自动演示 ──
    print()
    print("--- 演示 1：正常查询 ---")
    print(f"用户：帮我查一下订单 ORD001")
    print(f"客服：{enhanced_agent.handle_message('帮我查一下订单 ORD001', thread_id='demo')}")

    print()
    print("--- 演示 2：负面情绪 + 投诉（情绪预检 → 安抚 → 处理）---")
    print(f"用户：你们这破耳机质量太差了，我要投诉！马上退款！")
    print(f"客服：{enhanced_agent.handle_message('你们这破耳机质量太差了，我要投诉！马上退款！', thread_id='demo')}")

    print()
    print("--- 演示 3：FAQ 咨询（同一 demo 会话，验证多轮记忆）---")
    print(f"用户：退换货怎么操作？")
    print(f"客服：{enhanced_agent.handle_message('退换货怎么操作？', thread_id='demo')}")

    print()
    print("--- 演示 4：转人工场景 ---")
    print(f"用户：我要投诉你们公司，找你们经理来！")
    print(f"客服：{enhanced_agent.handle_message('我要投诉你们公司，找你们经理来！', thread_id='demo')}")

    print()
    print("=" * 55)
    print("   手动对话模式（输入 quit 退出）")
    print(f"   当前排队状态：{enhanced_agent.get_queue_status()}")
    print("=" * 55)
    print()

    session_id = "user_session_1"
    while True:
        try:
            user_input = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("感谢使用，再见！")
            break
        resp = enhanced_agent.handle_message(user_input, thread_id=session_id)
        print(f"小智：{resp}")
        print()
