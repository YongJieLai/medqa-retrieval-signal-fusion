"""构造 ground truth（Oracle）。

对每道题：
- 闭卷作答：不检索，模型直接选答案，记 correct_closed。
- 开卷作答：喂 gold rationale，模型再选答案，记 correct_open。
- 边际增益 Δ = correct_open - correct_closed（负值按 0 处理）。
- should_retrieve = 1 当且仅当 Δ == +1。

同时计算 M1（自评）与 M2（熟悉度）所需的模型侧信号，一起缓存，
避免重复调用 API（评测阶段纯本地可重跑）。

输出：results/ground_truth.json + results/llm_signals.json
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, ".")


def _load_env_file(path=".env"):
    """从 .env 文件读取 API key（不覆盖已有环境变量）。"""
    if os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        return
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k in ("DASHSCOPE_API_KEY", "OPENAI_API_KEY") and v:
                os.environ[k] = v


from src.data.synthetic_batch import build_synthetic_batch
from src.execute.llm_client import LLMClient


def main():
    _load_env_file()
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ 请先提供 API key（二选一）：")
        print("   1) 在 repo 根目录创建 .env 文件，写入：DASHSCOPE_API_KEY=sk-...")
        print("   2) export DASHSCOPE_API_KEY=sk-...")
        sys.exit(1)

    client = LLMClient(api_key=api_key)  # 默认 qwen-plus（DashScope）
    questions = build_synthetic_batch()

    os.makedirs("results", exist_ok=True)

    gt_rows = []
    signals = {}

    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] {q.question_id} ({q.qtype})")
        # 闭卷
        closed = client.answer_closed(q.stem, q.options)
        q.correct_closed = 1 if closed == q.answer else 0
        # 开卷
        open_ans = client.answer_open(q.stem, q.options, q.rationale)
        q.correct_open = 1 if open_ans == q.answer else 0
        # Δ（负值按 0）
        q.delta = max(0, q.correct_open - q.correct_closed)
        q.should_retrieve = 1 if q.delta == 1 else 0

        gt_rows.append(
            {
                "question_id": q.question_id,
                "qtype": q.qtype,
                "answer": q.answer,
                "closed_pred": closed,
                "open_pred": open_ans,
                "correct_closed": q.correct_closed,
                "correct_open": q.correct_open,
                "delta": q.delta,
                "should_retrieve": q.should_retrieve,
            }
        )

        # M1 自评 + M2 熟悉度（模型侧信号）
        try:
            self_eval = client.self_eval(q.stem, q.options)
        except Exception as e:
            print(f"   ⚠️ 自评失败：{e}，用默认 0.5")
            self_eval = 0.5
        # M2 熟悉度 MVP：用「闭卷是否答对」作为熟悉度代理（答对=熟悉=不需检索）。
        # DEVIATION: 完整版 KBM 用平均 token logprob，MVP 用答对代理以节省 API 调用。
        familiarity = 1.0 if q.correct_closed == 1 else 0.0
        signals[q.question_id] = {"self_eval": self_eval, "familiarity": familiarity}

    with open("results/ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(gt_rows, f, ensure_ascii=False, indent=2)
    with open("results/llm_signals.json", "w", encoding="utf-8") as f:
        json.dump(signals, f, ensure_ascii=False, indent=2)

    # 打印摘要
    n_pos = sum(1 for r in gt_rows if r["should_retrieve"] == 1)
    print("\n=== ground truth 摘要 ===")
    print(f"总题数: {len(gt_rows)}")
    print(f"类型1(罕见病): {sum(1 for r in gt_rows if r['qtype']=='type1')}")
    print(f"类型2(推理): {sum(1 for r in gt_rows if r['qtype']=='type2')}")
    print(f"should_retrieve=1 的题数: {n_pos}")
    print(f"闭卷平均正确率: {sum(r['correct_closed'] for r in gt_rows)/len(gt_rows):.2%}")
    print(f"开卷平均正确率: {sum(r['correct_open'] for r in gt_rows)/len(gt_rows):.2%}")


if __name__ == "__main__":
    main()
