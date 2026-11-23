"""qwen-flash 弱模型：H1 + 融合充分性验证。

目的：在 DashScope 唯一可用的弱模型 qwen-flash 上，重跑 ReDis-QA 真罕见子集，
     验证两件事：
     1. 弱模型是否"失明"（闭卷答错罕见病）—— H1 弱版本。
     2. 喂 ReCOP 聚焦证据后，检索增益（should_retrieve）是否 > 0 —— 融合充分性的前提。

关键改进（对比之前 should_retrieve≈0 的失败）：
     之前喂 Overview/Symptoms 泛化证据，答案（药物/机制/蛋白名）不在其中。
     这次根据题目考察类型（treatment / cause / diagnosis / symptom），
     针对性喂 Therapies / Causes / Diagnosis / Symptoms 证据，提高"直接含答案"概率。

输出：results/redisqa_flash_focused_gt.jsonl
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

sys.path.insert(0, ".")

from collections import defaultdict


def load_env():
    for line in open(".env", encoding="utf-8"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


from openai import OpenAI


MODEL = "qwen-flash-2025-07-28"


def load_recop():
    idx = defaultdict(dict)
    for line in open("data/recop.jsonl", encoding="utf-8"):
        r = json.loads(line)
        cat = r["title"].split(" of ")[0] if " of " in r["title"] else r["title"]
        idx[r["rare_disease"]][cat] = r["content"]
    return idx


def classify_question(question, correct_opt):
    """粗分类题目的考察类型，决定喂哪类 ReCOP 证据。"""
    q = question.lower()
    o = correct_opt.lower()
    text = q + " " + o
    if re.search(r"\bdrug\b|\btreat|\btherapy|\bmedication|\bmanagement\b|\bused in\b|\bprescrib", text):
        return "Therapies"
    if re.search(r"\bcause\b|\bdue to\b|\bdeficiency\b|\bmechanism\b|\bdefect\b|\bpathophysiolog|\bassociated with\b|\bresult", text):
        return "Causes"
    if re.search(r"\bdiagnos|\bconfirm|\bfinding\b|\binvestigation|\btest\b|\bexamin", text):
        return "Diagnosis"
    return "Symptoms"


def build_evidence(diseases, recop, category, max_len=1800):
    """按指定类别 + 兜底 Overview 构造证据。"""
    parts = []
    for d in diseases:
        if d not in recop:
            continue
        for cat in [category, "Overview"]:
            if cat in recop[d]:
                parts.append(f"【{d}·{cat}】{recop[d][cat]}")
                break  # 每病只取一类 + overview 兜底在下面单独加
    # 若太短，补 Overview
    if sum(len(p) for p in parts) < 300:
        for d in diseases:
            if d in recop and "Overview" in recop[d]:
                parts.append(f"【{d}·Overview】{recop[d]['Overview']}")
                break
    return "\n\n".join(parts)[:max_len]


def answer(question, opts, evidence=None):
    o = "\n".join(f"{chr(65+i)}. {x}" for i, x in enumerate(opts))
    if evidence:
        p = (f"你是医学多选题助手。结合下面给出的医学知识，选出唯一正确答案，只输出一个字母(A/B/C/D)。\n"
             f"题干：{question}\n选项：\n{o}\n相关知识：\n{evidence}\n答案：")
    else:
        p = (f"你是医学多选题助手。请只根据题干和选项选出唯一正确答案，只输出一个字母(A/B/C/D)。\n"
             f"题干：{question}\n选项：\n{o}\n答案：")
    for attempt in range(6):
        try:
            r = client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": p}],
                temperature=0.0, max_tokens=10)
            t = r.choices[0].message.content.strip().upper()
            for ch in t:
                if ch in "ABCD":
                    return ch
            return "?"
        except Exception as e:
            if "429" in str(e) or "limit" in str(e).lower() or "rate" in str(e).lower():
                time.sleep(8 * (2 ** attempt))
                continue
            raise
    return "?"


client = None


def main():
    global client
    load_env()
    client = OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"],
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    recop = load_recop()
    rows = [json.loads(l) for l in open("data/redisqa_truly_rare.jsonl", encoding="utf-8")]

    # 抽样 30 题（控制 API 调用量，60 次作答）
    import random
    rng = random.Random(42)
    sample = rng.sample(rows, 30)

    out = []
    n_closed_wrong = 0
    n_retrieve = 0
    t0 = time.time()

    for i, r in enumerate(sample):
        opts = [r["opa"], r["opb"], r["opc"], r["opd"]]
        ans = "ABCD"[r["cop"]]
        qid = f"redis_{i}"

        closed = answer(r["question"], opts)
        cc = int(closed == ans)
        if not cc:
            n_closed_wrong += 1

        # 聚焦证据
        cat = classify_question(r["question"], opts[r["cop"]])
        ev = build_evidence(r["rare_disease"], recop, cat)

        open_a = answer(r["question"], opts, ev)
        co = int(open_a == ans)
        delta = max(0, co - cc)
        if delta == 1:
            n_retrieve += 1

        out.append({
            "id": qid, "question": r["question"],
            "options": opts, "answer_idx": r["cop"],
            "rare_disease": r["rare_disease"],
            "closed_pred": closed, "open_pred": open_a,
            "correct_closed": cc, "correct_open": co,
            "delta": delta, "should_retrieve": delta,
            "evidence_category": cat,
            "evidence_len": len(ev),
        })

        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"[{i+1}/30] 闭卷错={n_closed_wrong} 检索救回={n_retrieve} 耗时={round(el,0)}s", flush=True)

    with open("results/redisqa_flash_focused_gt.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== qwen-flash 真罕见子集（聚焦证据，30 题）===")
    print(f"闭卷正确率: {30-n_closed_wrong}/30 = {(30-n_closed_wrong)/30:.0%}")
    print(f"开卷正确率: {sum(1 for r in out if r['correct_open']==1)}/30")
    print(f"闭卷答错: {n_closed_wrong}")
    print(f"should_retrieve=1(检索救回): {n_retrieve}")
    # 分类别看救回
    from collections import Counter
    print("证据类别分布:", Counter(r['evidence_category'] for r in out))
    print("各证据类别救回数:", Counter(r['evidence_category'] for r in out if r['should_retrieve']==1))


if __name__ == "__main__":
    main()
