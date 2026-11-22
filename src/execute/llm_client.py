"""执行层：调用 LLM 做闭卷/开卷作答。

代码纪律：模型冻结，全程同一个 backbone。
默认用通义千问（DashScope 兼容模式，国内可直连），也可切换到 gpt-4o-mini。
答题前约束：ground truth 计算中，闭卷 = 不喂知识，开卷 = 喂 gold 知识/rationale。
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from openai import OpenAI

# 千问（DashScope 兼容模式）默认端点
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# 默认模型：qwen-turbo（该 key 唯一开通的模型；qwen-plus/max/flash 均 403）
DEFAULT_MODEL = "qwen-turbo"


class LLMClient:
    """LLM 客户端（默认千问 DashScope，兼容 OpenAI 官方 API）。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        base_url: Optional[str] = None,
        use_dashscope: bool = True,
    ):
        if use_dashscope:
            self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
            self.base_url = base_url or DASHSCOPE_BASE_URL
        else:
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            self.base_url = base_url
        if not self.api_key:
            raise ValueError("缺少 API key：请设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY 环境变量")
        self.model = model
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

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
        resp = self._create_with_retry(prompt, max_tokens=10)
        text = resp.choices[0].message.content.strip()
        return self._parse_score(text)

    def _answer(self, prompt: str) -> str:
        resp = self._create_with_retry(prompt, max_tokens=10)
        text = resp.choices[0].message.content.strip().upper()
        # 提取第一个 A/B/C/D
        for ch in text:
            if ch in "ABCD":
                return ch
        return "A"  # 兜底（不应发生）

    def _create_with_retry(self, prompt: str, max_tokens: int = 10):
        """带限速重试的 chat.completions.create。

        应对 qwen-turbo 低 RPM 限流（429 limit_requests），
        指数退避重试，最多 5 次。
        """
        import time

        last_err = None
        for attempt in range(5):
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                last_err = e
                code = getattr(e, "code", None)
                status = getattr(e, "status_code", None)
                # 限流（429）→ 退避重试；其他错误直接抛
                if code == "limit_requests" or status == 429 or "rate" in str(e).lower():
                    wait = 20 * (2 ** attempt)  # 20s, 40s, 80s, 160s, 320s
                    print(f"      ⏳ 限流，{wait}s 后重试（第{attempt+1}次）...")
                    time.sleep(wait)
                    continue
                raise
        raise last_err

    @staticmethod
    def _parse_score(text: str) -> float:
        """从文本提取 0~1 分数。"""
        import re

        m = re.search(r"([0-9]*\.?[0-9]+)", text)
        if m:
            v = float(m.group(1))
            return max(0.0, min(1.0, v))
        return 0.5
