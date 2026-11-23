"""MedCoG 复现 · 第一步：三维元认知监测（Meta-Cognition Regulator 的 Monitoring 阶段）。

对 MedXpertQA P0 的 200 题，用 qwen-flash 打三个元认知分数：
- complexity（s_c）：是否需要多跳推理/逻辑分解
- familiarity（s_f）：是否教科书典型案例（能否复用历史经验）
- knowledge_density（s_k）：是否依赖具体医学事实（实体/关系）

familiarity 复用 LLMClient.familiarity()（logprob margin，读模型内部状态）；
complexity / knowledge_density 用 LLM 自评 prompt（0~1 分数）。

防泄漏：所有评分只用题干+选项，不碰 label。

输出：results/medcog_signals.jsonl（每题一行，含三维分数）
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, ".")

from openai import OpenAI

DATA_200 = "data/medxpertqa_p0_200.jsonl"
OUT = "results/medcog_signals.jsonl"
MODEL = "qwen-flash-2025-07-28"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def load_env():
    for line in open(".env", encoding="utf-8"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def call_llm(client, prompt, max_tokens=20, logprobs=False):
    """带重试的 LLM 调用。"""
    for attempt in range(6):
        try:
            return client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
                logprobs=logprobs,
                top_logprobs=5 if logprobs else None,
            )
        except Exception as e:
            if "429" in str(e) or "limit" in str(e).lower() or "rate" in str(e).lower():
                time.sleep(8 * (2 ** attempt))
                continue
            raise
    return None


def parse_score(text: str) -> float:
    """提取 0~1 分数。"""
    import re
    if text is None:
        return 0.5
    m = re.search(r"([0-9]*\.?[0-9]+)", text)
    if m:
        return max(0.0, min(1.0, float(m.group(1))))
    return 0.5


def build_options_str(options: dict) -> str:
    return "\n".join(f"({k}) {v}" for k, v in options.items())


def score_complexity(client, question, options):
    """s_c：复杂度。是否需要多跳推理/鉴别诊断/逻辑分解。"""
    opts = build_options_str(options)
    prompt = (
        "评估下面这道医学多选题的【推理复杂度】。\n"
        "复杂度指：作答是否需要多步推理、鉴别诊断、排除干扰选项，而非直接回忆单一事实。\n"
        "只输出一个 0 到 1 之间的数字（1=极高复杂度，需要多步鉴别诊断；0=直接回忆即可作答），不要解释。\n\n"
        f"题目：{question}\n\n选项：\n{opts}\n\n复杂度分数："
    )
    resp = call_llm(client, prompt, max_tokens=10)
    return parse_score(resp.choices[0].message.content if resp else None)


def score_knowledge_density(client, question, options):
    """s_k：知识密度。是否依赖具体医学事实（疾病/药物/机制等实体关系）。"""
    opts = build_options_str(options)
    prompt = (
        "评估下面这道医学多选题的【知识密度】。\n"
        "知识密度指：作答是否依赖具体的医学事实性知识（特定疾病、药物、生化机制、解剖实体及其关系），\n"
        "还是仅靠常识/通用临床判断即可。\n"
        "只输出一个 0 到 1 之间的数字（1=高度依赖精确医学事实；0=纯常识判断），不要解释。\n\n"
        f"题目：{question}\n\n选项：\n{opts}\n\n知识密度分数："
    )
    resp = call_llm(client, prompt, max_tokens=10)
    return parse_score(resp.choices[0].message.content if resp else None)


def score_familiarity(client, question, options):
    """s_f：熟悉度。用 logprob margin（最佳/次优 token 差距）作代理。

    margin 大 = 模型非常确定 = 熟悉；margin 小 = 犹豫 = 不熟悉。
    返回 0~1（1=熟悉）。这是读模型内部状态，不依赖答案对错。
    """
    import math
    opts = build_options_str(options)
    prompt = (
        "你是医学多选题助手。请只根据题干和选项选出唯一正确答案。\n"
        "严格只输出一个字母，不要输出任何解释。\n\n"
        f"题目：{question}\n\n选项：\n{opts}\n\n答案："
    )
    resp = call_llm(client, prompt, max_tokens=10, logprobs=True)
    if resp is None:
        return 0.5
    choice = resp.choices[0]
    if not hasattr(choice, "logprobs") or choice.logprobs is None:
        return 0.5
    lp = choice.logprobs
    if not hasattr(lp, "content") or not lp.content:
        return 0.5
    first = lp.content[0]
    if not hasattr(first, "top_logprobs") or not first.top_logprobs:
        return 0.5
    toplps = sorted([t.logprob for t in first.top_logprobs], reverse=True)
    if len(toplps) < 2:
        return 0.5
    margin = toplps[0] - toplps[1]
    familiarity = 1.0 / (1.0 + math.exp(-(margin - 5.0) / 5.0))
    return max(0.0, min(1.0, familiarity))


def main():
    load_env()
    client = OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url=BASE_URL)
    rows = [json.loads(l) for l in open(DATA_200, encoding="utf-8")]

    out = []
    t0 = time.time()
    for idx, r in enumerate(rows):
        qid = r["id"]
        question = r["question"]
        options = r["options"]

        s_c = score_complexity(client, question, options)
        s_f = score_familiarity(client, question, options)
        s_k = score_knowledge_density(client, question, options)

        rec = {
            "id": qid,
            "question_type": r["question_type"],
            "medical_task": r["medical_task"],
            "body_system": r["body_system"],
            "label": r["label"],
            "complexity": round(s_c, 4),
            "familiarity": round(s_f, 4),
            "knowledge_density": round(s_k, 4),
        }
        out.append(rec)

        if (idx + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  [{idx+1}/{len(rows)}] 耗时 {round(el,0)}s  "
                  f"(c={s_c:.2f}, f={s_f:.2f}, k={s_k:.2f})", flush=True)

    os.makedirs("results", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n已保存 {len(out)} 条到 {OUT}", flush=True)


if __name__ == "__main__":
    main()
