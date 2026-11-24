"""50 条场景评测：规则命中判定 + 指标报告

用法（仓库根目录）：
    .\\venv\\Scripts\\python.exe eval.py               # 全量 50 条
    .\\venv\\Scripts\\python.exe eval.py --limit 3     # 冒烟测试
    .\\venv\\Scripts\\python.exe eval.py --cat 查订单   # 只跑某一类

说明：
- 每条用例使用独立 thread_id，记忆互不污染；跨轮用例才共享同一会话。
- 判定是「工具命中 + 关键词命中」的必要条件，不等于人工评审的"正确"。
  它适合看趋势、定位回归，不适合对外宣称准确率。
"""

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import main
from kb import delete_doc, ingest_file
from main import EnhancedCustomerServiceAgent, init_db, init_faq

# ================================================================
# 用例定义
#   tools      : 必须全部调用到
#   tools_any  : 至少调用到其中一个
#   any_of     : 回复里至少出现一个关键词
#   all_of     : 回复里必须全部出现
#   forbid_any : 回复里不能出现
#   transfer   : True / False / None（None = 不判定）
# ================================================================
CASES = [
    # ── 订单与物流（10）──
    {"id": "order-01", "cat": "查订单", "turns": ["帮我查一下订单 ORD077"],
     "tools": ["query_order"], "any_of": ["ORD077"], "transfer": False},
    {"id": "order-02", "cat": "查订单", "turns": ["查下 ORD001 到哪了"],
     "tools": ["query_order"], "any_of": ["ORD001"], "transfer": False},
    # 设计上：订单不存在 → 工具空结果 → 按规则转人工（外壳会替换回复）
    {"id": "order-03", "cat": "查订单", "turns": ["我的订单 ORD999 怎么样了"],
     "tools": ["query_order"], "transfer": True},
    {"id": "order-04", "cat": "查订单", "turns": ["我想查订单"],
     "any_of": ["订单号"], "transfer": None},
    {"id": "order-05", "cat": "查订单", "turns": ["帮我查 ORD105", "这个订单是什么状态"],
     "tools": ["query_order"], "any_of": ["ORD105"], "transfer": False},
    {"id": "order-06", "cat": "查订单", "turns": ["ORD042 买的是什么商品"],
     "tools": ["query_order"], "any_of": ["ORD042"], "transfer": False},
    {"id": "order-07", "cat": "查订单", "turns": ["我上周买的耳机到哪了"],
     "any_of": ["订单号"], "transfer": None},
    {"id": "order-08", "cat": "查订单", "turns": ["帮我看看 ORD088 的金额"],
     "tools": ["query_order"], "any_of": ["ORD088"], "transfer": False},
    {"id": "order-09", "cat": "查订单", "turns": ["查一下 ORD007"],
     "tools": ["query_order"], "any_of": ["ORD007"], "transfer": False},
    {"id": "order-10", "cat": "查订单", "turns": ["ORD050 现在什么状态", "那能取消吗"],
     "tools": ["query_order"], "any_of": ["ORD050"], "transfer": False},

    # ── FAQ 政策咨询（12）──
    {"id": "faq-01", "cat": "FAQ咨询", "turns": ["怎么申请退换货"],
     "tools": ["search_faq"], "any_of": ["7 天", "7天", "七天"], "transfer": False},
    {"id": "faq-02", "cat": "FAQ咨询", "turns": ["退款多久能到账"],
     "tools": ["search_faq"], "any_of": ["3-5", "3~5", "3 到 5", "工作日"], "transfer": False},
    {"id": "faq-03", "cat": "FAQ咨询", "turns": ["发货后多久能收到"],
     "tools": ["search_faq"], "any_of": ["2-5", "2~5", "2 到 5", "工作日"], "transfer": False},
    {"id": "faq-04", "cat": "FAQ咨询", "turns": ["可以修改收货地址吗"],
     "tools": ["search_faq"], "any_of": ["未发货", "订单号", "地址"], "transfer": False},
    {"id": "faq-05", "cat": "FAQ咨询", "turns": ["蓝牙耳机怎么连接"],
     "tools": ["search_faq"], "any_of": ["配对", "蓝牙", "电源键"], "transfer": False},
    {"id": "faq-06", "cat": "FAQ咨询", "turns": ["商品有质量问题怎么办"],
     "tools": ["search_faq"], "any_of": ["订单号", "照片", "售后"], "transfer": False},
    {"id": "faq-07", "cat": "FAQ咨询", "turns": ["你们客服几点上班"],
     "tools": ["search_faq"], "any_of": ["9:00", "9点", "22:00", "22点"], "transfer": False},
    {"id": "faq-08", "cat": "FAQ咨询", "turns": ["退货运费谁承担"],
     "tools": ["search_faq"], "any_of": ["运费", "买家", "商家"], "transfer": False},
    {"id": "faq-09", "cat": "FAQ咨询", "turns": ["七天无理由退货需要什么条件"],
     "tools": ["search_faq"], "any_of": ["完好", "二次销售", "7 天", "七天"], "transfer": False},
    {"id": "faq-10", "cat": "FAQ咨询", "turns": ["怎么开发票"],
     "tools_any": ["search_faq", "search_kb"], "any_of": ["发票"], "transfer": False},
    {"id": "faq-11", "cat": "FAQ咨询", "turns": ["订单一般多久发货"],
     "tools": ["search_faq"], "any_of": ["24 小时", "24小时", "48 小时", "现货"], "transfer": False},
    {"id": "faq-12", "cat": "FAQ咨询", "turns": ["支持货到付款吗"],
     "tools": ["search_faq"], "any_of": ["货到付款"], "transfer": False},

    # ── 情绪安抚（8，判定情绪规则层是否生效）──
    {"id": "emo-01", "cat": "情绪", "turns": ["你们这什么垃圾服务，气死我了，马上给我解决"],
     "any_of": ["非常抱歉给您带来"], "transfer": None},
    {"id": "emo-02", "cat": "情绪", "turns": ["太差了，骗子，赶紧处理"],
     "any_of": ["非常抱歉给您带来"], "transfer": None},
    {"id": "emo-03", "cat": "情绪", "turns": ["我很着急，帮我查一下 ORD015 到哪了"],
     "tools": ["query_order"], "any_of": ["ORD015"], "transfer": False},
    {"id": "emo-04", "cat": "情绪", "turns": ["我要投诉你们客服"],
     "any_of": ["很抱歉给您添麻烦"], "transfer": None},
    {"id": "emo-05", "cat": "情绪", "turns": ["气死我了，你们这就是坑人，马上退款！"],
     "any_of": ["非常抱歉给您带来"], "transfer": None},
    {"id": "emo-06", "cat": "情绪", "turns": ["你好，帮我查下 ORD003"],
     "tools": ["query_order"], "any_of": ["ORD003"],
     "forbid_any": ["非常抱歉给您带来", "很抱歉给您添麻烦"], "transfer": False},
    {"id": "emo-07", "cat": "情绪", "turns": ["垃圾客服，曝光你们，12315 投诉"],
     "any_of": ["很抱歉给您添麻烦"], "transfer": None},
    {"id": "emo-08", "cat": "情绪", "turns": ["有点急，帮我查 ORD060"],
     "tools": ["query_order"], "any_of": ["ORD060"], "transfer": False},

    # ── 转人工（8，判定风险兜底链路）──
    {"id": "tr-01", "cat": "转人工", "turns": ["我要退款，还要你们赔偿我的损失"],
     "transfer": True},
    {"id": "tr-02", "cat": "转人工", "turns": ["我要起诉你们，这是欺诈"],
     "transfer": True},
    {"id": "tr-03", "cat": "转人工", "turns": ["我的账号被盗了，别人登录我的账号"],
     "transfer": True},
    {"id": "tr-04", "cat": "转人工", "turns": ["有人把我的个人信息泄露了，我要投诉"],
     "transfer": True},
    {"id": "tr-05", "cat": "转人工", "turns": ["我要十倍赔偿，不然就曝光你们"],
     "transfer": True},
    {"id": "tr-06", "cat": "转人工", "turns": ["我要向消协投诉你们"],
     "transfer": True},
    {"id": "tr-07", "cat": "转人工", "turns": ["我要找你们领导，这事没完"],
     "transfer": True},
    {"id": "tr-08", "cat": "转人工", "turns": ["我要退款到别人的支付宝账户"],
     "transfer": True},

    # ── 知识库检索（8，需加载 eval_assets/客服政策.md）──
    {"id": "kb-01", "cat": "知识库", "turns": ["发票多久能收到"],
     "tools_any": ["search_kb", "search_faq"], "any_of": ["3 个工作日", "3个工作日", "发票"], "transfer": False},
    {"id": "kb-02", "cat": "知识库", "turns": ["保修期是多久"],
     "tools_any": ["search_kb", "search_faq"], "any_of": ["12 个月", "12个月", "一年", "保修"], "transfer": False},
    {"id": "kb-03", "cat": "知识库", "turns": ["你们支持以旧换新吗"],
     "tools_any": ["search_kb", "search_faq"], "any_of": ["以旧换新"], "transfer": False},
    {"id": "kb-04", "cat": "知识库", "turns": ["双十一有什么活动"],
     "tools_any": ["search_kb", "search_faq"], "any_of": ["满 300", "满300", "满减", "活动"], "transfer": False},
    {"id": "kb-05", "cat": "知识库", "turns": ["耳机进水了还保修吗"],
     "tools_any": ["search_kb", "search_faq"], "any_of": ["进水", "不在保修", "不保修", "人为"], "transfer": False},
    {"id": "kb-06", "cat": "知识库", "turns": ["能开增值税专用发票吗"],
     "tools_any": ["search_kb", "search_faq"], "any_of": ["增值税", "专票", "专用发票"], "transfer": False},
    {"id": "kb-07", "cat": "知识库", "turns": ["旧手机怎么估价"],
     "tools_any": ["search_kb", "search_faq"], "any_of": ["估价", "48 小时", "48小时", "检测"], "transfer": False},
    {"id": "kb-08", "cat": "知识库", "turns": ["我的订单数据会保留多久"],
     "tools_any": ["search_kb", "search_faq"], "any_of": ["3 年", "3年", "三年"], "transfer": False},

    # ── 边界与幻觉（4）──
    # 同上：查不到订单时外壳会转人工，回复里只剩排队信息
    {"id": "bd-01", "cat": "边界", "turns": ["帮我查订单 ORD8888"],
     "tools": ["query_order"], "any_of": ["人工"], "transfer": True},
    {"id": "bd-02", "cat": "边界", "turns": ["帮我编一个 ORD999 的物流信息"],
     "any_of": ["无法", "不能", "不会", "不提供"], "transfer": None},
    {"id": "bd-03", "cat": "边界", "turns": ["你们公司老板是谁"],
     "any_of": ["无法", "不清楚", "人工", "客服"], "transfer": None},
    {"id": "bd-04", "cat": "边界", "turns": ["帮我写一首诗"],
     "any_of": ["客服", "订单", "无法", "帮您"], "transfer": None},
]

TRANSFER_KW = ("转接人工", "转人工", "为您转接", "排队序号")

# 当前用例调用过的工具（由 trace_chat 写入）
_CALLED: set[str] = set()


def _traced_chat(user_input: str, thread_id: str = "default") -> str:
    """替换 main.chat：调用同一个 Agent，但记录本轮用了哪些工具。"""
    agent = main.get_agent()
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 10}
    result = agent.invoke({"messages": [("user", user_input)]}, config=config)
    for msg in result["messages"]:
        for call in getattr(msg, "tool_calls", None) or []:
            _CALLED.add(call.get("name"))
    return result["messages"][-1].content


# EnhancedCustomerServiceAgent 内部调用的是 main 模块里的 chat，替换后即可被记录
main.chat = _traced_chat


def _is_transfer(reply: str) -> bool:
    return any(kw in reply for kw in TRANSFER_KW)


def grade(case: dict, reply: str) -> list[str]:
    """返回失败原因列表；空列表 = 通过。"""
    fails: list[str] = []

    if case.get("tools"):
        missing = [t for t in case["tools"] if t not in _CALLED]
        if missing:
            fails.append(f"未调用工具 {missing}")
    if case.get("tools_any") and not set(case["tools_any"]) & _CALLED:
        fails.append(f"未调用任一工具 {case['tools_any']}")
    if case.get("any_of") and not any(s in reply for s in case["any_of"]):
        fails.append("回复缺少期望信息")
    if case.get("all_of"):
        missing = [s for s in case["all_of"] if s not in reply]
        if missing:
            fails.append(f"回复缺少 {missing}")
    if case.get("forbid_any"):
        hit = [s for s in case["forbid_any"] if s in reply]
        if hit:
            fails.append(f"出现不应有的内容 {hit}")
    if case.get("transfer") is not None and _is_transfer(reply) != case["transfer"]:
        fails.append("转人工判定不符")

    return fails


def _pctl(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * p))))
    return ordered[idx]


def run(args) -> None:
    init_db()
    print("[准备] 初始化 FAQ 向量库（增量同步 50 条，首次加载嵌入模型约 30 秒）...")
    init_faq()

    cases = CASES
    if args.cat:
        cases = [c for c in cases if c["cat"] == args.cat]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print(f"[错误] 没有匹配的用例（--cat {args.cat!r}），可用类别：")
        print("  " + "、".join(sorted({c["cat"] for c in CASES})))
        return

    doc_id = None
    if any(c["cat"] == "知识库" for c in cases):
        asset = Path("eval_assets") / "客服政策.md"
        if asset.exists():
            info = ingest_file(asset)
            doc_id = info["doc_id"]
            print(f"[准备] 评测知识库已载入：{asset.name}（{info['chunks']} 段）")
        else:
            print(f"[警告] 未找到 {asset}，跳过知识库用例")
            cases = [c for c in cases if c["cat"] != "知识库"]

    agent = EnhancedCustomerServiceAgent()
    results: list[dict] = []
    latencies: list[float] = []

    try:
        for i, case in enumerate(cases, 1):
            _CALLED.clear()
            reply, error, case_lat = "", None, []

            for msg in case["turns"]:
                started = time.perf_counter()
                try:
                    reply = agent.handle_message(msg, thread_id=f"eval-{case['id']}")
                except Exception as exc:  # noqa: BLE001 - 评测需要记录任何异常
                    error = f"{type(exc).__name__}: {exc}"
                    break
                case_lat.append(time.perf_counter() - started)

            latencies.extend(case_lat)
            fails = [error] if error else grade(case, reply)
            results.append({
                "id": case["id"],
                "cat": case["cat"],
                "turns": case["turns"],
                "reply": reply,
                "tools": sorted(_CALLED),
                "fails": fails,
                "ok": not fails,
                "latency_s": round(sum(case_lat), 2),
            })

            mark = "PASS" if not fails else "FAIL"
            detail = "；".join(fails) if fails else ""
            print(f"[{i:02d}/{len(cases)}] {mark} {case['id']:<9} {detail}")
            if args.delay:
                time.sleep(args.delay)
    finally:
        if doc_id:
            delete_doc(doc_id)
            print("[清理] 评测知识库文档已从向量库移除")

    report(results, latencies, cases)


def report(results: list[dict], latencies: list[float], cases: list[dict]) -> None:
    total = len(results)
    passed = sum(1 for r in results if r["ok"])

    tool_cases = [r for r, c in zip(results, cases) if c.get("tools") or c.get("tools_any")]
    tool_ok = sum(1 for r in tool_cases if not any("工具" in f for f in r["fails"]))

    tr_cases = [r for r, c in zip(results, cases) if c.get("transfer") is not None]
    tr_ok = sum(1 for r in tr_cases if not any("转人工" in f for f in r["fails"]))

    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in results:
        by_cat[r["cat"]][1] += 1
        if r["ok"]:
            by_cat[r["cat"]][0] += 1

    def pct(a: int, b: int) -> str:
        return f"{a / b * 100:.1f}%" if b else "-"

    print("\n" + "=" * 62)
    print("评测汇总")
    print("=" * 62)
    print(f"用例总数          ：{total}")
    print(f"整体通过率        ：{passed}/{total}（{pct(passed, total)}）")
    print(f"工具调用命中率    ：{tool_ok}/{len(tool_cases)}（{pct(tool_ok, len(tool_cases))}）")
    print(f"转人工判定准确率  ：{tr_ok}/{len(tr_cases)}（{pct(tr_ok, len(tr_cases))}）")
    if latencies:
        print(f"平均响应时延      ：{statistics.mean(latencies):.2f}s")
        print(f"P95 响应时延      ：{_pctl(latencies, 0.95):.2f}s")
    print("-" * 62)
    for cat, (ok, n) in sorted(by_cat.items(), key=lambda kv: -kv[1][1]):
        print(f"  {cat:<6} {ok}/{n}  {pct(ok, n)}")
    print("=" * 62)

    failed = [r for r in results if not r["ok"]]
    if failed:
        print("\n失败明细（供排查，不等于全部是产品缺陷）：")
        for r in failed:
            print(f"  - {r['id']}：{'；'.join(r['fails'])}")

    summary = {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0,
        "tool_hit_rate": round(tool_ok / len(tool_cases), 4) if tool_cases else None,
        "transfer_acc": round(tr_ok / len(tr_cases), 4) if tr_cases else None,
        "latency_avg_s": round(statistics.mean(latencies), 3) if latencies else None,
        "latency_p95_s": round(_pctl(latencies, 0.95), 3) if latencies else None,
        "by_category": {k: {"passed": v[0], "total": v[1]} for k, v in by_cat.items()},
    }
    out = Path("eval_results.json")
    out.write_text(
        json.dumps({"summary": summary, "cases": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n明细已写入 {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="电商客服 Agent 评测")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟测试）")
    parser.add_argument("--cat", default="", help="只跑某一类，如：查订单 / FAQ咨询 / 情绪 / 转人工 / 知识库 / 边界")
    parser.add_argument("--delay", type=float, default=0.2, help="用例间隔秒数，避免触发限流")
    run(parser.parse_args())
