"""coverage 公式的单元测试（代码纪律 §4：公式函数必须配单测，解析手算校验）。

验证：
1. sigmoid 数值稳定（正负输入都对）。
2. idf 公式 idf(c) = log(N/(1+df(c)))。
3. 缺失概率 p_c = σ((idf - idf_median)/τ)。
4. coverage = Σ p_c · ν_c。
"""
import math
import sys

sys.path.insert(0, ".")

from src.signal.idf import sigmoid, idf_from_df, missing_prob, median
from src.signal.coverage import CoverageSignal


def approx(a, b, eps=1e-9):
    return abs(a - b) < eps


def test_sigmoid():
    # σ(0) = 0.5
    assert approx(sigmoid(0.0), 0.5)
    # σ(-10) ≈ 0.0000454
    assert approx(sigmoid(-10.0), 1.0 / (1.0 + math.exp(10)), 1e-6)
    # σ(10) ≈ 0.9999546
    assert approx(sigmoid(10.0), math.exp(-10) / (1.0 + math.exp(-10)) if False else 1.0 / (1.0 + math.exp(-10)), 1e-6)
    # 对称性：σ(x) + σ(-x) = 1
    assert approx(sigmoid(3.0) + sigmoid(-3.0), 1.0)


def test_idf():
    # N=100, df=0 -> log(100/1) = log(100)
    assert approx(idf_from_df(0, 100), math.log(100))
    # N=100, df=99 -> log(100/100)=0
    assert approx(idf_from_df(99, 100), 0.0)
    # N=100, df=24 -> log(100/25)=log(4)
    assert approx(idf_from_df(24, 100), math.log(4))


def test_median():
    assert median([1, 2, 3]) == 2.0
    assert approx(median([1, 2, 3, 4]), 2.5)
    assert median([]) == 0.0


def test_missing_prob():
    # idf == median -> σ(0) = 0.5
    assert approx(missing_prob(2.0, 2.0, 1.0), 0.5)
    # idf > median -> p_c > 0.5
    assert missing_prob(3.0, 2.0, 1.0) > 0.5
    # idf < median -> p_c < 0.5
    assert missing_prob(1.0, 2.0, 1.0) < 0.5


def test_coverage():
    # 手算样例：两个概念，idf 分别为 3.0 和 1.0，median=2.0，τ=1.0
    # p1 = σ(1.0) ≈ 0.73106, p2 = σ(-1.0) ≈ 0.26894
    # ν1=1, ν2=0（只有概念1 是必经节点）
    # coverage = 0.73106 * 1 + 0.26894 * 0 = 0.73106
    idf_map = {"c1": 3.0, "c2": 1.0}
    sig = CoverageSignal(idf_map=idf_map, tau=1.0)

    from src.schema import Question
    q = Question(question_id="q", stem="s", options=["a", "b", "c", "d"], qtype="type1", concepts=["c1", "c2"])
    kg_paths = {"c1": {"p1"}, "c2": set()}  # c1 在唯一路径，c2 不在

    expected_p1 = sigmoid(1.0)
    got = sig.coverage(q, kg_paths)
    assert approx(got, expected_p1), f"expected {expected_p1}, got {got}"

    # 两个都是必经节点：coverage = p1 + p2 = σ(1)+σ(-1) = 1
    kg_paths2 = {"c1": {"p1"}, "c2": {"p2"}}
    got2 = sig.coverage(q, kg_paths2)
    assert approx(got2, 1.0), f"expected 1.0, got {got2}"


def test_missing_prob_tau_effect():
    # τ 越大，缺失概率越接近 0.5（越平滑）
    p_small_tau = missing_prob(4.0, 2.0, 0.5)
    p_large_tau = missing_prob(4.0, 2.0, 2.0)
    assert p_small_tau > p_large_tau


if __name__ == "__main__":
    test_sigmoid()
    test_idf()
    test_median()
    test_missing_prob()
    test_coverage()
    test_missing_prob_tau_effect()
    print("✅ 全部单元测试通过")
