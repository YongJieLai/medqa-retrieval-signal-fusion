"""DeepSeek backbone 的 CoT 能力对照实验。

目的：验证「换 DeepSeek 是否因为其推理（CoT）能力更强，从而在 MedXpertQA 上更有价值」。

对照 4 个配置（200 题，温度 0）：
- deepseek-chat    × Direct（直接答）
- deepseek-chat    × CoT（显式逐步推理）
- deepseek-reasoner × Direct（推理模型自带内部 CoT）
- deepseek-reasoner × CoT（外部 CoT prompt + 内置推理）

关键观察：
1. chat 的 CoT 是否 >= Direct（千问曾出现 CoT < Direct，即 10.5% < 17.5%）
2. reasoner 是否整体强于 chat
3. reasoner 上「外部 CoT」是否还有额外增益

输出：results/deepseek_cot_ablation.jsonl
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from openai import OpenAI

DATA_200 = "data/medxpertqa_p0_200.jsonl"
OUT = "results/deepseek_cot_ablation.jsonl"

DEEPSEEK_URL = "https://api.deepseek.com"

FORMAT_INSTRUCTION = (
    "Output your answer in exactly this format:\n"
    "Answer: <LETTER>\n"
    "Reason: <one-sentence justification>"
)


def load_env():
    for line in open(".env", encoding="utf-8"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def build_direct_prompt(question, options):
    opts = "\n".join(f"({k}) {v}" for k, v in sorted(options.items()))
    return (
        "You are an expert physician. Answer the following multiple-choice medical question.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{opts}\n\n"
        f"{FORMAT_INSTRUCTION}"
    )


def build_cot_prompt(question, options):
    opts = "\n".join(f"({k}) {v}" for k, v in sorted(options.items()))
    return (
        "You are an expert physician. Solve the following multiple-choice medical question "
        "step by step: first list the key clinical findings and your differential reasoning, "
        "then select the best answer.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{opts}\n\n"
        f"{FORMAT_INSTRUCTION}"
    )


def parse_answer(text: str) -> str:
    m = re.search(r"Answer\s*:\s*([A-Ja-j])", text)
    if m:
        return m.group(1).upper()
    for ch in reversed(text.upper()):
        if ch in "ABCDEFGHIJ":
            return ch
    return "?"


def call_llm(client, model, prompt, max_tokens=800):
    """返回 (content, finish_reason, reasoning_tokens)。reasoner 用大 max_tokens。"""
    mt = max_tokens
    if model == "deepseek-reasoner":
        mt = 16384  # 推理模型需要大 token 预算，避免 reasoning 截断导致无答案
    for attempt in range(6):
        try:
            r = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=mt)
            msg = r.choices[0].message
            reasoning_tokens = 0
            try:
                reasoning_tokens = r.usage.completion_tokens_details.reasoning_tokens or 0
            except Exception:
                pass
            return (msg.content or "").strip(), r.choices[0].finish_reason, reasoning_tokens
        except Exception as e:
            s = str(e)
            if "429" in s or "rate" in s.lower() or "limit" in s.lower():
                time.sleep(6 * (2 ** attempt))
                continue
            raise
    return "", "error", 0


def main():
    load_env()
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=DEEPSEEK_URL)
    rows = [json.loads(l) for l in open(DATA_200, encoding="utf-8")]

    configs = [
        ("chat", "Direct"),
        ("chat", "CoT"),
        ("reasoner", "Direct"),
        ("reasoner", "CoT"),
    ]
    model_map = {"chat": "deepseek-chat", "reasoner": "deepseek-reasoner"}

    out = []
    t0 = time.time()
    for idx, r in enumerate(rows):
        qid = r["id"]
        question, options, label = r["question"], r["options"], r["label"]
        rec = {
            "id": qid,
            "question_type": r["question_type"],
            "medical_task": r["medical_task"],
            "label": label,
        }

        for kind, mode in configs:
            model = model_map[kind]
            prompt = build_direct_prompt(question, options) if mode == "Direct" \
                else build_cot_prompt(question, options)
            text, finish, reasoning_tokens = call_llm(client, model, prompt)
            rec[f"{kind}_{mode}"] = parse_answer(text)
            rec[f"{kind}_{mode}_finish"] = finish
            rec[f"{kind}_{mode}_reasoning_tokens"] = reasoning_tokens

        out.append(rec)

        if (idx + 1) % 10 == 0:
            el = time.time() - t0
            n = len(out)
            line = f"  [{idx+1}/{len(rows)}] {round(el,0)}s"
            for kind, mode in configs:
                acc = sum(1 for x in out if x[f"{kind}_{mode}"] == x["label"]) / n
                line += f"  {kind}-{mode}={acc:.3f}"
            print(line, flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    n = len(out)
    print(f"\n===== 最终准确率（N={n}）=====", flush=True)
    for kind, mode in configs:
        acc = sum(1 for x in out if x[f"{kind}_{mode}"] == x["label"]) / n
        print(f"  {kind}-{mode}: {acc:.4f}", flush=True)

    # 关键对照
    chat_direct = sum(1 for x in out if x["chat_Direct"] == x["label"]) / n
    chat_cot = sum(1 for x in out if x["chat_CoT"] == x["label"]) / n
    reasoner_direct = sum(1 for x in out if x["reasoner_Direct"] == x["label"]) / n
    reasoner_cot = sum(1 for x in out if x["reasoner_CoT"] == x["label"]) / n
    print(f"\n  chat  CoT-Direct = {chat_cot - chat_direct:+.4f}", flush=True)
    print(f"  reasoner CoT-Direct = {reasoner_cot - reasoner_direct:+.4f}", flush=True)
    print(f"  reasoner-Direct - chat-Direct = {reasoner_direct - chat_direct:+.4f}", flush=True)
    print(f"\n已保存 {n} 条到 {OUT}", flush=True)


if __name__ == "__main__":
    main()
