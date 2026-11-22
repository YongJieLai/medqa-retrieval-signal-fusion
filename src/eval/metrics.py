"""评测层：四个核心指标（分题型报告）。

- 漏检率 Miss Rate：should_retrieve=1 的题里，被判「不查」的比例 → 越小越好。
- 误检率 False Alarm：should_retrieve=0 的题里，被判「查」的比例 → 越小越好。
- top-B 命中率 Hit@B：选出的 B 道题里 should_retrieve=1 的占比 → 越大越好。
- 端到端准确率 Acc@B：选中的 B 道检索、其余闭卷，全批次最终正确率 → 越大越好。
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from ..schema import Decision, LabeledQuestion


def evaluate(
    questions: Sequence[LabeledQuestion],
    decisions: Sequence[Decision],
) -> Dict[str, float]:
    """对一批题 + 对应决策算四个指标（不分类别，整体）。"""
    qid2q = {q.question_id: q for q in questions}
    # 分 should_retrieve=1 / 0 两组
    pos = [q for q in questions if q.should_retrieve == 1]
    neg = [q for q in questions if q.should_retrieve == 0]
    qid2dec = {d.question_id: d for d in decisions}

    def action(q: LabeledQuestion) -> str:
        return qid2dec[q.question_id].action

    # 漏检率：pos 里被判 closed 的比例
    miss = sum(1 for q in pos if action(q) == "closed") / len(pos) if pos else 0.0
    # 误检率：neg 里被判 retrieve 的比例
    fa = sum(1 for q in neg if action(q) == "retrieve") / len(neg) if neg else 0.0
    # Hit@B：被判 retrieve 的题里 should_retrieve=1 的占比
    retrieved = [q for q in questions if action(q) == "retrieve"]
    hit = sum(1 for q in retrieved if q.should_retrieve == 1) / len(retrieved) if retrieved else 0.0
    # Acc@B：retrieve 的题用 correct_open，closed 的题用 correct_closed
    acc = (
        sum(q.correct_open if action(q) == "retrieve" else q.correct_closed for q in questions)
        / len(questions)
        if questions
        else 0.0
    )
    return {
        "miss_rate": miss,
        "false_alarm": fa,
        "hit_at_b": hit,
        "acc_at_b": acc,
        "n": len(questions),
        "n_retrieve": len(retrieved),
    }


def evaluate_by_type(
    questions: Sequence[LabeledQuestion],
    decisions: Sequence[Decision],
) -> Dict[str, Dict[str, float]]:
    """按题型（type1 / type2）分别评测。"""
    result = {}
    for qtype in ("type1", "type2"):
        sub_q = [q for q in questions if q.qtype == qtype]
        if not sub_q:
            continue
        sub_d = [d for d in decisions if d.question_id in {q.question_id for q in sub_q}]
        result[qtype] = evaluate(sub_q, sub_d)
    result["all"] = evaluate(list(questions), list(decisions))
    return result
