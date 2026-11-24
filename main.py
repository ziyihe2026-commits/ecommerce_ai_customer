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
import hashlib
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

FAQ_SEED: list[tuple[str, str]] = [
    # ── 订单与物流 ──
    ("如何查询我的订单？", "请提供您的订单号（如 ORD001），我立刻帮您查询。"),
    ("订单多久发货？", "现货商品在付款后 24 小时内发出，预售商品以商品页标注时间为准。"),
    ("怎么查看物流信息？", "发货后可在订单详情页查看物流单号，或把订单号发给我，我帮您查询当前状态。"),
    ("可以修改收货地址吗？", "订单未发货前可修改地址，请提供订单号和新地址。"),
    ("可以修改订单里的商品吗？", "订单提交后不支持直接更换商品，建议取消后重新下单。"),
    ("如何取消订单？", "待付款订单可直接取消；已发货订单请联系客服申请拦截。"),
    ("订单显示已发货但一直没收到怎么办？", "请提供订单号，我帮您核查物流；超过 7 天无更新将为您升级处理。"),
    ("支持货到付款吗？", "目前支持在线支付和货到付款，具体以结算页显示的支付方式为准。"),
    ("可以开发票吗？", "支持电子发票，下单时勾选开票，或联系客服补开。"),
    ("发票什么时候能收到？", "电子发票在订单完成后 3 个工作日内发送到您预留的邮箱。"),

    # ── 退换货与售后 ──
    ("怎么申请退换货？", "收到商品 7 天内可申请退换货。请提供订单号，我为您生成退换货申请链接。"),
    ("退款什么时候到账？", "退款将在仓库签收退货后 3-5 个工作日内原路返回您的支付账户。"),
    ("商品有质量问题怎么办？", "非常抱歉！请提供订单号和商品照片，我立即帮您走售后流程。"),
    ("七天无理由退货的条件是什么？", "商品需保持完好、不影响二次销售，配件和赠品齐全，并在签收后 7 天内申请。"),
    ("退货运费谁承担？", "质量问题的退货运费由我们承担；七天无理由退货的运费由买家承担。"),
    ("换货需要多久？", "仓库签收后 1-2 个工作日发出新品，运输时效与正常发货一致。"),
    ("拆封使用过还能退吗？", "不影响二次销售的前提下可以退；已使用导致商品贬损的，按实际情况折价处理。"),
    ("赠品需要一起退回吗？", "需要。赠品缺失会影响退款金额，请随主商品一并寄回。"),
    ("如何申请仅退款？", "未发货订单可直接申请仅退款；已发货订单需协商后由客服操作。"),
    ("商品少件或发错货怎么办？", "请提供订单号和开箱照片，核实后我们为您补发或退款。"),
    ("保修期多久？", "大部分商品提供 12 个月质保，具体以商品页保修说明为准。"),
    ("衣服尺码不合适可以换吗？", "可以。吊牌完好、未洗涤的情况下，7 天内可申请换码。"),

    # ── 商品与使用 ──
    ("蓝牙耳机怎么连接？", "长按电源键 3 秒进入配对模式，手机蓝牙搜索设备名 BT-XXX 点击连接即可。"),
    ("机械键盘怎么切换灯光模式？", "按 Fn + Ins 切换灯效，Fn + 方向键调节亮度。"),
    ("充电宝能带上飞机吗？", "额定能量不超过 100Wh 可随身携带，不可托运。"),
    ("显示器支架怎么安装？", "请按说明书先固定底座再挂装显示器，承重不超过 9kg。"),
    ("商品有异味怎么办？", "新品密封包装可能有轻微气味，通风 24 小时后可散去；若持续存在请申请售后。"),
    ("怎么辨别是不是正品？", "每件商品附有防伪码，可在品牌官网或官方公众号查询验证。"),
    ("电池怎么保养？", "避免长期满电或完全放电，建议电量保持在 20%-80% 之间。"),
    ("耳机单边没声音怎么排查？", "先在另一台设备测试，再检查耳塞清洁和蓝牙重连；仍异常可申请换新。"),
    ("支持以旧换新吗？", "部分数码商品支持以旧换新，具体以商品页活动标识为准。"),
    ("商品参数和页面不符怎么办？", "请提供实拍照片，核实属实后支持退货并由我们承担运费。"),

    # ── 支付与账户 ──
    ("支持哪些支付方式？", "支持微信、支付宝、银行卡及花呗或白条分期。"),
    ("优惠券怎么使用？", "结算页选择可用优惠券，每单限用一张，不可与部分活动叠加。"),
    ("积分可以抵扣吗？", "100 积分抵 1 元，单笔订单最高抵扣订单金额的 10%。"),
    ("如何修改绑定手机号？", "在账户安全页验证原手机号后更换；原号码不可用请联系人工客服。"),
    ("会员权益有哪些？", "会员享专属折扣、免运费券和优先客服，具体权益见会员中心。"),
    ("价格保护怎么申请？", "签收后 7 天内商品降价可申请补差，需提供降价截图。"),
    ("支持分期付款吗？", "支持 3/6/12 期分期，具体期数和手续费以结算页为准。"),
    ("账户异常登录怎么办？", "请立即修改密码并联系人工客服冻结账户。"),

    # ── 服务与时间 ──
    ("客服工作时间是什么？", "在线客服工作时间：每天 9:00-22:00。其余时间可留言，次日回复。"),
    ("夜间下单什么时候处理？", "夜间订单从次日 9:00 起按付款顺序依次处理。"),
    ("人工客服怎么联系？", "在对话框输入“转人工”，或拨打客服热线 400-000-0000。"),
    ("可以预约配送时间吗？", "部分区域支持预约配送，下单时在配送选项中勾选即可。"),

    # ── 隐私与合规 ──
    ("我的个人信息如何保护？", "我们仅将信息用于订单履约和售后服务，不会向第三方出售。"),
    ("不满意处理结果怎么投诉？", "请提供订单号和问题描述，我们将升级至客服主管并在 24 小时内回复。"),
    ("可以删除我的账户数据吗？", "可以。提交注销申请后将在 15 个工作日内清除个人数据。"),
    ("消费投诉走什么流程？", "可拨打 12315 或通过全国 12315 平台提交，我们也会全力配合处理。"),
    ("你们会给我发营销短信吗？", "仅在您授权后发送，可随时在消息设置中退订。"),
    ("退货后个人信息会保留吗？", "仅保留法律法规要求保存的交易记录，其余信息按隐私政策处理。"),
]


def _faq_id(question: str) -> str:
    """用问题文本生成稳定 ID，供增量同步使用"""
    return "faq-" + hashlib.md5(question.encode("utf-8")).hexdigest()[:12]


def init_faq():
    """构建 FAQ 向量库（增量幂等：只同步新增/删除，不重复追加）"""
    global FAQ_DATA, FAQ_COLLECTION

    FAQ_DATA.clear()
    FAQ_DATA.extend({"question": q, "answer": a} for q, a in FAQ_SEED)

    FAQ_COLLECTION = Chroma(
        collection_name="faq",
        embedding_function=get_embeddings(),
        persist_directory="./chroma_faq",
    )

    questions = [item["question"] for item in FAQ_DATA]
    ids = [_faq_id(q) for q in questions]
    existing = set(FAQ_COLLECTION.get()["ids"])

    # 删除已从种子中移除的问题
    stale = [i for i in existing if i not in set(ids)]
    if stale:
        FAQ_COLLECTION.delete(ids=stale)

    # 只写入新增的问题（答案从 FAQ_DATA 按 question 反查，改答案无需重建向量）
    new = [(i, q) for i, q in zip(ids, questions) if i not in existing]
    if new:
        FAQ_COLLECTION.add_texts(texts=[q for _, q in new], ids=[i for i, _ in new])


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

            # ── ④ 判断是否需要转人工（话术追加在原回复之后，不丢答案）──
            if self._need_human_agent(response):
                if thread_id in self.transferred_threads:
                    # 已转过人工，不再重复排队，直接把模型回复返回
                    return soothing + response
                return f"{soothing}{response}\n\n{self._transfer_to_human(user_input, thread_id)}"

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
        # 不要匹配裸的"人工客服"：正常回答里的"如需人工客服…"会被误判成转人工。
        # （评测 faq-07 暴露过：答案被替换成排队话术）
        keywords = ["转人工", "转接人工", "无法处理", "无法直接处理"]
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
