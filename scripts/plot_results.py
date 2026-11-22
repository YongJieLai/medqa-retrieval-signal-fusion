"""绘图：两类题 × 各方法漏检率的柱状图（用于论文 Figure 1 motivating example）。

读取 results/core_table.csv，生成 grouped bar chart：
x 轴 = 方法，每组两根柱 = 类型1（罕见病）/ 类型2（推理）的漏检率。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# 中文字体
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    df = pd.read_csv("results/core_table.csv")
    # 只取 type1 / type2（去掉 all）
    df = df[df["qtype"].isin(["type1", "type2"])]

    methods = ["coverage(ours)", "M1_MedCoG自评", "M2_KBM熟悉度", "M3_Marina", "M4_QuCo-RAG", "random"]
    method_labels = ["coverage\n(ours)", "MedCoG\n自评", "KBM\n熟悉度", "Marina\n流行度", "QuCo-RAG\n共现", "random"]

    type1_vals = []
    type2_vals = []
    for m in methods:
        row1 = df[(df["method"] == m) & (df["qtype"] == "type1")]
        row2 = df[(df["method"] == m) & (df["qtype"] == "type2")]
        type1_vals.append(row1["miss_rate"].iloc[0] if len(row1) else 0)
        type2_vals.append(row2["miss_rate"].iloc[0] if len(row2) else 0)

    import numpy as np
    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, type1_vals, width, label="类型1：罕见病（高缺失×低推理）", color="#c0392b")
    bars2 = ax.bar(x + width / 2, type2_vals, width, label="类型2：推理题（低缺失×高推理）", color="#3498db")

    ax.set_ylabel("漏检率（Miss Rate，越小越好）")
    ax.set_title("混合长尾批次下各方法的漏检率（按题型）")
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 标注数值
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    os.makedirs("results", exist_ok=True)
    fig.tight_layout()
    fig.savefig("results/miss_rate_by_type.png", dpi=150)
    print("✅ 图已保存到 results/miss_rate_by_type.png")


if __name__ == "__main__":
    main()
