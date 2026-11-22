"""用真实 logprobs 计算各方法的模型侧信号（无 ground truth 泄漏）。

M1 自评：模型自评「是否需要检索」分数。
M2 熟悉度：用模型作答的 token logprob 算 perplexity 熟悉度（真实内部信号）。

输出：results/model_signals_real.json
{
  "<question_id>": {"self_eval": float, "familiarity": float}
}
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, ".")

from src.execute.llm_client import LLMClient


def load_env():
    for line in open(".env", encoding="utf-8"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def main():
    load_env()
    client = LLMClient()

    gt_rows = [json.loads(l) for l in open("results/mixed_batch_gt.jsonl", encoding="utf-8")]

    signals = {}
    for i, r in enumerate(gt_rows):
        opts = [r["options"][k] for k in "ABCD"]
        qid = r["id"]
        try:
            se = client.self_eval(r["question"], opts)
        except Exception as e:
            print(f"  ⚠️ {qid} 自评失败: {e}")
            se = 0.5
        try:
            fam = client.familiarity(r["question"], opts)
        except Exception as e:
            print(f"  ⚠️ {qid} 熟悉度失败: {e}")
            fam = 0.5
        signals[qid] = {"self_eval": se, "familiarity": fam}
        if (i + 1) % 10 == 0:
            print(f"进度 {i+1}/{len(gt_rows)}")

    with open("results/model_signals_real.json", "w", encoding="utf-8") as f:
        json.dump(signals, f, ensure_ascii=False, indent=2)

    print(f"\n完成，共 {len(signals)} 题的模型侧信号已保存")
    # 打印熟悉度分布，看区分度
    fams = [s["familiarity"] for s in signals.values()]
    print(f"熟悉度: 均值={sum(fams)/len(fams):.3f} 最小={min(fams):.3f} 最大={max(fams):.3f}")


if __name__ == "__main__":
    main()
