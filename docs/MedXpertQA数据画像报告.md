# MedXpertQA 数据画像与字段校验报告（P0）

> 评测计划书 P0 阶段产出。执行日期：2026-11-23
> 数据：`data/medxpertqa_text.jsonl`（全量 Text 子集）

---

## 1. 数据源确认

| 项 | 值 |
|---|---|
| 来源 | `TsinghuaC3I/MedXpertQA`（ICML 2025，arXiv:2501.18362） |
| 子集 | Text（纯文本，10 选 1），**不含 MM 图像子集** |
| 本地路径 | `data/medxpertqa_text.jsonl` |
| 总题数 | **2450**（计划书预估约 1800，实际为 Text 子集全量） |
| label 可用性 | **全部 2450 题均含 label**（无隐藏，可直接评测） |

> 结论：无需重新下载，本地已有完整带 label 的 Text 子集。计划书 §2.2 下载步骤可跳过。

## 2. 字段结构（单条示例）

```json
{
  "id": "Text-817",
  "question": "A 37-year-old male ... (题干全文)",
  "options": {"A": "...", "B": "...", ..., "J": "..."},
  "label": "H",
  "medical_task": "Diagnosis",
  "body_system": "Digestive",
  "question_type": "Reasoning"
}
```

字段说明：`id`（题目编号）、`question`（题干）、`options`（10 个选项 A-J）、`label`（正确选项）、`medical_task`（任务类型）、`body_system`（身体系统）、`question_type`（题型）。

## 3. 分布画像

### 3.1 题型（question_type）—— 最重要的分层变量

| 类型 | 题数 | 占比 |
|---|---|---|
| Reasoning | 1861 | 75.96% |
| Understanding | 589 | 24.04% |

### 3.2 任务类型（medical_task）

| 任务 | 题数 | 占比 |
|---|---|---|
| Diagnosis | 1050 | 42.86% |
| Treatment | 746 | 30.45% |
| Basic Science | 654 | 26.69% |

### 3.3 身体系统（body_system，12 类）

| 系统 | 题数 |
|---|---|
| Nervous | 386 |
| Skeletal | 355 |
| Cardiovascular | 306 |
| Digestive | 274 |
| Reproductive | 201 |
| Respiratory | 193 |
| Muscular | 177 |
| Endocrine | 176 |
| Other / NA | 126 |
| Urinary | 123 |
| Lymphatic | 85 |
| Integumentary | 48 |

### 3.4 题型 × 任务交叉

| question_type | medical_task | 题数 |
|---|---|---|
| Reasoning | Diagnosis | 921 |
| Reasoning | Treatment | 631 |
| Understanding | Basic Science | 345 |
| Reasoning | Basic Science | 309 |
| Understanding | Diagnosis | 129 |
| Understanding | Treatment | 115 |

### 3.5 其他统计

- **答案字母分布**：A-J 基本均匀（222~263 题/字母），无明显偏置。
- **题干长度**：mean=1094，min=191，max=4527（题干较长，含完整临床场景）。

## 4. 评测抽样方案

- **采用分层抽样 200 题**（`data/medxpertqa_p0_200.jsonl`，random_state=42）：
  - Reasoning 152 / Understanding 48（比例对齐全集的 76%/24%）
  - Diagnosis 86 / Treatment 61 / Basic Science 53
  - body_system 覆盖 12 类
- **该抽样已含 `query` 字段**（去选项后的题干主体），用于防泄漏检索（不碰答案选项）。
- **6 个方法共用同一份 200 题**，保证可比性。

## 5. 数据泄漏风险声明

MedXpertQA 是 2025 年公开数据，所用 LLM（qwen-plus 等）可能已见过部分题目。此为已知风险（计划书 §11.3），报告中已声明，无法完全排除。

---

## 6. 结论

数据质量合格，字段完整，label 全部可用，分层抽样已就绪。**可直接进入 P1（M0 + M1 跑通）**。
