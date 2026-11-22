"""执行层：调用 gpt-4o-mini 做闭卷/开卷作答。

代码纪律：模型冻结，全程同一个 backbone（本实验 MVP 用 gpt-4o-mini）。
答题前约束：ground truth 计算中，闭卷 = 不喂知识，开卷 = 喂 gold 知识/rationale。
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from openai import OpenAI


class LLMClient:
    """gpt-4o-mini 客户端（OpenAI 官方 API）。"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini", base_url: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("缺少 OPENAI_API_KEY：请设置环境变量或传入 api_key")
        self.model = model
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)

    def _build_prompt(self, stem: str, options: List[str]) -> str:
        opts = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(options))
        return (
            "你是医学多选题助手。请只根据题干和选项选出唯一正确答案。\n"
            "严格只输出一个字母（A/B/C/D），不要输出任何解释。\n\n"
            f"题干：{stem}\n\n选项：\n{opts}\n\n答案："
        )

    def answer_closed(self, stem: str, options: List[str]) -> str:
        """闭卷作答：不喂任何知识。返回 A/B/C/D。"""
        return self._answer(self._build_prompt(stem, options))

    def answer_open(self, stem: str, options: List[str], rationale: str) -> str:
        """开卷作答：喂 gold 知识/rationale。返回 A/B/C/D。"""
        prompt = self._build_prompt(stem, options) + "\n"
        prompt = (
            "你是医学多选题助手。请结合下面给出的医学知识，选出唯一正确答案。\n"
            "严格只输出一个字母（A/B/C/D），不要输出任何解释。\n\n"
            f"题干：{stem}\n\n选项：\n"
            + "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(options))
            + f"\n\n相关知识：{rationale}\n\n答案："
        )
        return self._answer(prompt)

    def self_eval(self, stem: str, options: List[str]) -> float:
        """M1 · MedCoG 自评：让模型输出 0~1 的「是否需要检索」分数。

        1 = 强烈需要检索（不确定），0 = 完全确定。"""
        opts = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(options))
        prompt = (
            "对下面这道医学多选题，评估「你需要额外检索医学资料才能答对」的程度。\n"
            "只输出一个 0 到 1 之间的数字（1=强烈需要检索，0=完全不需要），不要解释。\n\n"
            f"题干：{stem}\n\n选项：\n{opts}\n\n检索需求分数："
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        text = resp.choices[0].message.content.strip()
        return self._parse_score(text)

    def _answer(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        text = resp.choices[0].message.content.strip().upper()
        # 提取第一个 A/B/C/D
        for ch in text:
            if ch in "ABCD":
                return ch
        return "A"  # 兜底（不应发生）

    @staticmethod
    def _parse_score(text: str) -> float:
        """从文本提取 0~1 分数。"""
        import re

        m = re.search(r"([0-9]*\.?[0-9]+)", text)
        if m:
            v = float(m.group(1))
            return max(0.0, min(1.0, v))
        return 0.5
