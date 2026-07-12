from dotenv import load_dotenv
import os
from langgraph.prebuilt import create_react_agent
from langchain_deepseek import ChatDeepSeek
import sqlite3
from langchain.tools import tool

# 加载环境变量
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# 1. 初始化数据库（订单表）
def init_db():
    conn = sqlite3.connect("order.db")
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS orders (
        order_id TEXT PRIMARY KEY,
        user_name TEXT,
        goods_name TEXT,
        price REAL,
        status TEXT
    )
    ''')
    # 插入测试订单数据
    test_data = [
        ("ORD001", "张三", "无线蓝牙耳机", 199.0, "已发货"),
        ("ORD002", "李四", "机械键盘", 129.0, "待付款"),
        ("ORD003", "王五", "电竞鼠标", 89.0, "已完成")
    ]
    cursor.executemany("INSERT OR IGNORE INTO orders VALUES (?,?,?,?,?)", test_data)
    conn.commit()
    conn.close()

# 2. 定义查询订单工具
@tool
def query_order(order_id: str) -> str:
    """
    根据订单号查询用户订单信息
    Args:
        order_id: 订单编号，如 ORD001
    """
    conn = sqlite3.connect("order.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,))
    res = cursor.fetchone()
    conn.close()
    if res:
        return f"订单号：{res[0]}，用户：{res[1]}，商品：{res[2]}，价格：{res[3]}，状态：{res[4]}"
    else:
        return "未查询到该订单，请检查订单号是否正确"

# 3. 初始化大模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY
)

# 4. 构建React Agent
tools = [query_order]
agent_graph = create_react_agent(llm, tools)

# 5. 对话测试函数
def chat_customer_service(user_input: str):
    response = agent_graph.invoke({"messages": [("user", user_input)]})
    return response["messages"][-1].content

if __name__ == "__main__":
    # 初始化数据库
    init_db()
    print("==== 电商智能客服Agent已启动 ====")
    print("测试对话1：查询ORD001订单")
    print(chat_customer_service("帮我查一下订单ORD001"))
    print("-" * 50)
    print("测试对话2：查询ORD999不存在订单")
    print(chat_customer_service("查订单ORD999"))
    print("-" * 50)
    print("测试对话3：咨询商品相关问题")
    print(chat_customer_service("蓝牙耳机质量怎么样？"))