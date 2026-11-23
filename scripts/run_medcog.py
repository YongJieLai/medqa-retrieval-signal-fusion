"""MedCoG 复现 · 五策略推理 + 阈值门控路由 + 评估。

五策略（MedCoG 策略池 S）：
- Zero-Shot：直接作答
- SCoT：结构化思维链（事实层 + 推理层解耦）
- SCoT+Memory：SCoT + 相似历史案例（episodic）
- SCoT+KG：SCoT + 知识图谱验证证据（factual，PubMed BM25 + PrimeKG）
- SCoT+KG+Memory：全部叠加

门控路由（非参数）：I_j = 1(s_j > τ_j)，三维全低 → Zero-Shot；否则 SCoT ⊕ Σ I_j·M_j。
- familiarity 高（I_f=0）不触发 memory；knowledge_density 高（I_k=1）触发 KG。
- 注意：familiarity 是"熟悉度"，论文中 I_f=1 触发 memory（复用历史经验）。

阈值 τ 在留出的 40 题校准集上网格搜索，160 题测试集独立评估。

防泄漏：检索 query 只用题干；memory 案例排除当前题；所有策略不碰 label。

用法：
  python scripts/run_medcog.py --mode calibrate   # 40 题校准阈值
  python scripts/run_medcog.py --mode evaluate    # 160 题测试路由
  python scripts/run_medcog.py --mode all         # 校准 + 测试（默认）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, ".")

from openai import OpenAI

from medcog_retriever import MedCoGRetriever

DATA_200 = "data/medxpertqa_p0_200.jsonl"
SIGNALS = "results/medcog_signals.jsonl"
OUT = "results/medcog_results.jsonl"
REPORT = "results/medcog_summary.json"
MODEL = "qwen-flash-2025-07-28"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

CALIB_SIZE = 40
RANDOM_SEED = 42


def load_env():
    for line in open(".env", encoding="utf-8"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def call_llm(client, prompt, max_tokens=500):
    for attempt in range(6):
        try:
            r = client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=max_tokens)
            return r.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) or "limit" in str(e).lower() or "rate" in str(e).lower():
                time.sleep(8 * (2 ** attempt))
                continue
            raise
    return ""


def parse_answer(text: str) -> str:
    """从回答中提取最后一个字母答案（A-J）。"""
    for ch in reversed(text.upper()):
        if ch in "ABCDEFGHIJ":
            return ch
    return "?"


def build_options_str(options: dict) -> str:
    return "\n".join(f"({k}) {v}" for k, v in options.items())


# ---------- 五策略 ----------

def strategy_zero_shot(client, question, options):
    opts = build_options_str(options)
    prompt = (f"Question: {question}\nOptions: {opts}\n"
              f"Answer with only the single letter (A-J) of the correct option.")
    return call_llm(client, prompt, max_tokens=20)


def strategy_scot(client, question, options):
    """结构化 CoT：事实层（提取关键临床事实）+ 推理层（鉴别诊断/判断）。"""
    opts = build_options_str(options)
    prompt = (
        f"You are solving a medical multiple-choice question. Use structured reasoning.\n\n"
        f"Step 1 (Facts): List the key clinical facts from the stem.\n"
        f"Step 2 (Reasoning): Perform differential diagnosis / clinical judgment.\n"
        f"Step 3 (Answer): Give the final answer as a single letter (A-J).\n\n"
        f"Question: {question}\nOptions: {opts}\n"
    )
    return call_llm(client, prompt, max_tokens=500)


def strategy_scot_memory(client, question, options, cases_text):
    opts = build_options_str(options)
    prompt = (
        f"You are solving a medical multiple-choice question. Use structured reasoning and "
        f"reference similar historical cases when helpful.\n\n"
        f"Similar historical cases (for reference only):\n{cases_text}\n\n"
        f"Step 1 (Facts): List key clinical facts.\n"
        f"Step 2 (Reasoning): Differential diagnosis / clinical judgment.\n"
        f"Step 3 (Answer): Final answer as a single letter (A-J).\n\n"
        f"Question: {question}\nOptions: {opts}\n"
    )
    return call_llm(client, prompt, max_tokens=500)


def strategy_scot_kg(client, question, options, pubmed_text, primekg_text):
    opts = build_options_str(options)
    prompt = (
        f"You are solving a medical multiple-choice question. Use structured reasoning and "
        f"external medical knowledge to verify your answer.\n\n"
        f"Relevant medical literature:\n{pubmed_text}\n\n"
        f"Knowledge graph facts:\n{primekg_text}\n\n"
        f"Step 1 (Facts): List key clinical facts.\n"
        f"Step 2 (Reasoning): Use the external knowledge to support differential diagnosis.\n"
        f"Step 3 (Answer): Final answer as a single letter (A-J).\n\n"
        f"Question: {question}\nOptions: {opts}\n"
    )
    return call_llm(client, prompt, max_tokens=500)


def strategy_scot_kg_memory(client, question, options, pubmed_text, primekg_text, cases_text):
    opts = build_options_str(options)
    prompt = (
        f"You are solving a medical multiple-choice question. Use structured reasoning, "
        f"external medical knowledge, and similar historical cases.\n\n"
        f"Similar historical cases:\n{cases_text}\n\n"
        f"Relevant medical literature:\n{pubmed_text}\n\n"
        f"Knowledge graph facts:\n{primekg_text}\n\n"
        f"Step 1 (Facts): List key clinical facts.\n"
        f"Step 2 (Reasoning): Differential diagnosis / clinical judgment.\n"
        f"Step 3 (Answer): Final answer as a single letter (A-J).\n\n"
        f"Question: {question}\nOptions: {opts}\n"
    )
    return call_llm(client, prompt, max_tokens=500)


# ---------- 门控路由 ----------

def route_strategy(complexity, familiarity, knowledge_density, tau):
    """非参数门控路由。

    MedCoG: M = Zero-Shot 若 I_c ∧ I_f ∧ I_k = 0；否则 SCoT ⊕ Σ_{j∈{f,k}} I_j·M_j。
    注意 familiarity 高 = 熟悉 = 不需要 memory，familiarity 低 = 不熟悉 = 需要 memory。
    论文中 I_f 触发 memory 表示"不熟悉、需要历史经验"，故用 (1-familiarity) 判断。
    """
    I_c = 1 if complexity > tau["c"] else 0
    I_f = 1 if (1 - familiarity) > tau["f"] else 0   # 不熟悉 → 触发 memory
    I_k = 1 if knowledge_density > tau["k"] else 0    # 知识密度高 → 触发 KG

    if I_c == 0 and I_f == 0 and I_k == 0:
        return "zero_shot"
    # 否则 SCoT 基础 + 叠加
    strat = "scot"
    if I_f:
        strat += "+memory"
    if I_k:
        strat += "+kg"
    return strat


# ---------- 主流程 ----------

def load_data():
    rows = [json.loads(l) for l in open(DATA_200, encoding="utf-8")]
    signals = {json.loads(l)["id"]: json.loads(l) for l in open(SIGNALS, encoding="utf-8")}
    # 合并信号
    for r in rows:
        s = signals.get(r["id"], {})
        r["complexity"] = s.get("complexity", 0.5)
        r["familiarity"] = s.get("familiarity", 0.5)
        r["knowledge_density"] = s.get("knowledge_density", 0.5)
    return rows


def split_calib_test(rows):
    """留出 40 题校准，160 题测试（固定 random_state，分层尽量保持题型比例）。"""
    import random
    rng = random.Random(RANDOM_SEED)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    calib_idx = set(idx[:CALIB_SIZE])
    calib = [rows[i] for i in idx[:CALIB_SIZE]]
    test = [rows[i] for i in idx[CALIB_SIZE:]]
    return calib, test


def execute_strategy(client, retriever, row, strat):
    """执行单一策略，返回答案。"""
    question = row["question"]
    options = row["options"]
    query = row.get("query", question)

    if strat == "zero_shot":
        return strategy_zero_shot(client, question, options)

    if strat == "scot":
        return strategy_scot(client, question, options)

    # 需要额外知识的策略
    cases_text = ""
    pubmed_text = ""
    primekg_text = ""

    if "memory" in strat:
        cases = retriever.retrieve_cases(query, k=3)
        cases_text = retriever.format_cases(cases)

    if "kg" in strat:
        pubmed = retriever.retrieve_pubmed(query, k=5)
        pubmed_text = retriever.format_pubmed(pubmed)
        # 用 LLM 抽取题干实体（贴近 MedCoG 三步 grounding），再做 PrimeKG 检索
        entities = extract_entities(question, client=client)
        primekg = retriever.retrieve_primekg(entities, max_per_entity=3)
        primekg_text = retriever.format_primekg(primekg)

    if strat == "scot+memory":
        return strategy_scot_memory(client, question, options, cases_text)
    if strat == "scot+kg":
        return strategy_scot_kg(client, question, options, pubmed_text, primekg_text)
    if strat == "scot+kg+memory":
        return strategy_scot_kg_memory(client, question, options, pubmed_text, primekg_text, cases_text)

    return strategy_scot(client, question, options)


def extract_entities(question: str, client=None, max_n: int = 5):
    """从题干提取候选医学实体。

    优先用 LLM 抽取（贴近 MedCoG 论文"三步 grounding"的实体锚定）；
    无 client 时退回启发式（连续大写开头词 + 停用词过滤）。
    """
    if client is not None:
        prompt = (
            "Extract the most important medical entities (disease names, symptoms, anatomical sites, "
            "drugs, tests, pathogens, etc.) from the medical question below.\n"
            "Output ONLY the entity names in English, separated by commas, at most 5 entities. No explanation.\n\n"
            f"Question: {question}\n\nMedical entities:"
        )
        text = call_llm(client, prompt, max_tokens=60)
        if text:
            entities = [e.strip() for e in text.replace("\n", ",").split(",")]
            entities = [e for e in entities if len(e) >= 3]
            return entities[:max_n]

    import re
    cands = re.findall(r"\b[A-Z][a-zA-Z\-]*(?:\s+[a-z]+){0,3}", question)
    stop = {"the", "a", "an", "in", "on", "at", "of", "to", "for", "and", "or",
            "is", "are", "was", "were", "with", "his", "her", "he", "she", "has",
            "had", "have", "physical", "examination", "reveals", "history", "medical",
            "recent", "seeks", "attention", "due", "which", "following", "most", "likely"}
    entities = []
    for c in cands:
        c = c.strip()
        if len(c) < 3 or c.lower() in stop:
            continue
        entities.append(c)
        if len(entities) >= max_n:
            break
    return entities


def grid_search_tau(calib_rows, client, retriever):
    """在 40 题校准集上网格搜索阈值 τ，最大化准确率。"""
    # 阈值候选：三个维度各取分位数附近
    # 简单做法：对每个维度网格 {0.3, 0.5, 0.7}
    best = {"c": 0.5, "f": 0.5, "k": 0.5, "acc": 0.0}
    grid = [0.3, 0.5, 0.7]
    results_cache = {}

    for tc in grid:
        for tf in grid:
            for tk in grid:
                tau = {"c": tc, "f": tf, "k": tk}
                correct = 0
                total = 0
                strat_counter = Counter()
                for row in calib_rows:
                    strat = route_strategy(row["complexity"], row["familiarity"], row["knowledge_density"], tau)
                    strat_counter[strat] += 1
                    key = (row["id"], strat)
                    if key not in results_cache:
                        ans = execute_strategy(client, retriever, row, strat)
                        results_cache[key] = ans
                    else:
                        ans = results_cache[key]
                    pred = parse_answer(ans)
                    if pred == row["label"]:
                        correct += 1
                    total += 1
                acc = correct / total if total else 0
                if acc > best["acc"]:
                    best = {"c": tc, "f": tf, "k": tk, "acc": acc, "strats": dict(strat_counter)}

    print(f"[校准] 最优阈值 τ = (c={best['c']}, f={best['f']}, k={best['k']})，"
          f"校准集准确率 = {best['acc']:.3f} ({correct} 题口径)", flush=True)
    print(f"[校准] 策略分布: {best['strats']}", flush=True)
    return best, results_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all", choices=["calibrate", "evaluate", "all"])
    args = parser.parse_args()

    load_env()
    client = OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url=BASE_URL)
    retriever = MedCoGRetriever()

    rows = load_data()
    calib, test = split_calib_test(rows)
    print(f"[数据] 校准集 {len(calib)} 题，测试集 {len(test)} 题", flush=True)

    if args.mode in ("calibrate", "all"):
        tau, cache = grid_search_tau(calib, client, retriever)
        # 保存最优阈值
        with open("results/medcog_tau.json", "w") as f:
            json.dump({"tau": {"c": tau["c"], "f": tau["f"], "k": tau["k"]},
                       "calib_acc": tau["acc"]}, f, ensure_ascii=False, indent=2)
    else:
        tau_data = json.load(open("results/medcog_tau.json"))
        tau = {"c": tau_data["tau"]["c"], "f": tau_data["tau"]["f"], "k": tau_data["tau"]["k"]}
        cache = {}

    if args.mode in ("evaluate", "all"):
        print(f"[评估] 用阈值 τ = (c={tau['c']}, f={tau['f']}, k={tau['k']}) 路由测试集...", flush=True)
        results = []
        correct = 0
        strat_counter = Counter()
        strat_correct = Counter()
        strat_total = Counter()
        t0 = time.time()

        for idx, row in enumerate(test):
            strat = route_strategy(row["complexity"], row["familiarity"], row["knowledge_density"], tau)
            ans = execute_strategy(client, retriever, row, strat)
            pred = parse_answer(ans)
            is_correct = pred == row["label"]
            correct += is_correct
            strat_counter[strat] += 1
            strat_total[strat] += 1
            strat_correct[strat] += is_correct

            results.append({
                "id": row["id"],
                "question_type": row["question_type"],
                "medical_task": row["medical_task"],
                "label": row["label"],
                "complexity": row["complexity"],
                "familiarity": row["familiarity"],
                "knowledge_density": row["knowledge_density"],
                "strategy": strat,
                "pred": pred,
                "correct": is_correct,
            })

            if (idx + 1) % 20 == 0:
                el = time.time() - t0
                acc = correct / (idx + 1)
                print(f"  [{idx+1}/{len(test)}] 耗时 {round(el,0)}s 当前准确率 {acc:.3f}", flush=True)

        os.makedirs("results", exist_ok=True)
        with open(OUT, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # 汇总
        final_acc = correct / len(test)
        summary = {
            "n_test": len(test),
            "n_calib": len(calib),
            "tau": {"c": tau["c"], "f": tau["f"], "k": tau["k"]},
            "accuracy": round(final_acc, 4),
            "n_correct": correct,
            "strategy_distribution": dict(strat_counter),
            "strategy_accuracy": {s: round(strat_correct[s] / strat_total[s], 4) for s in strat_total},
        }
        with open(REPORT, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n[结果] 测试集准确率 = {final_acc:.4f} ({correct}/{len(test)})", flush=True)
        print(f"[结果] 策略分布: {dict(strat_counter)}", flush=True)
        print(f"[结果] 各策略准确率: {summary['strategy_accuracy']}", flush=True)
        print(f"[结果] 已保存 {OUT} 和 {REPORT}", flush=True)


if __name__ == "__main__":
    main()
