"""reference.py 自体の sanity check + fixture の自己整合性チェック。

JS 側との数値一致は tests/js/test_matrix.test.mjs で検証する。ここではあくまで
「numpy 参照実装が pathological になっていないか」「JSON 化しても shape が
保たれるか」「gelu(0)=0 のような自明な恒等式が成り立つか」を確認する。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

# このパッケージから reference を import するためのパス調整。
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from reference import (  # noqa: E402
    fix_matmul, fix_softmax, fix_layernorm, fix_gelu,
    _gelu_tanh, _layernorm, _softmax,
)


# ─── matmul ─────────────────────────────────────────────────────────────

def test_matmul_shape():
    fx = fix_matmul(3, 4, 5, seed=0)
    assert fx["M"] == 3 and fx["K"] == 4 and fx["N"] == 5
    assert len(fx["A"]) == 3 and len(fx["A"][0]) == 4
    assert len(fx["B"]) == 4 and len(fx["B"][0]) == 5
    assert len(fx["C"]) == 3 and len(fx["C"][0]) == 5


def test_matmul_seed_repeatable():
    a = fix_matmul(4, 5, 6, seed=0)
    b = fix_matmul(4, 5, 6, seed=0)
    assert a["A"] == b["A"]
    assert a["C"] == b["C"]


def test_matmul_different_seeds_differ():
    a = fix_matmul(4, 5, 6, seed=0)
    b = fix_matmul(4, 5, 6, seed=1)
    # 少なくとも A の最初の要素は異なるはず
    assert a["A"][0][0] != b["A"][0][0]


# ─── softmax ────────────────────────────────────────────────────────────

def test_softmax_rows_sum_to_one():
    fx = fix_softmax(4, 7, seed=0)
    for row in fx["y"]:
        assert math.isclose(sum(row), 1.0, abs_tol=1e-12)


def test_softmax_handles_large_scale():
    """scale を大きくしても overflow せず、全行が確率分布になる。"""
    fx = fix_softmax(3, 8, seed=2, scale=5.0)
    for row in fx["y"]:
        assert math.isclose(sum(row), 1.0, abs_tol=1e-12)
        for v in row:
            assert 0.0 <= v <= 1.0


def test_softmax_extreme_values_stable():
    """[1000, 1001, 999] のような極端な値でも overflow しない (max 引き)。"""
    x = np.array([[1000.0, 1001.0, 999.0]])
    y = _softmax(x, axis=-1)
    assert math.isclose(y.sum(), 1.0, abs_tol=1e-15)
    assert all(np.isfinite(y).flatten())


# ─── LayerNorm ──────────────────────────────────────────────────────────

def test_layernorm_shape():
    fx = fix_layernorm(5, 16, seed=0)
    assert len(fx["x"]) == 5
    assert len(fx["x"][0]) == 16
    assert len(fx["gamma"]) == 16
    assert len(fx["beta"]) == 16
    assert len(fx["y"]) == 5
    assert len(fx["y"][0]) == 16


def test_layernorm_finite():
    """LayerNorm の出力に NaN/Inf が混じらないこと。"""
    fx = fix_layernorm(5, 16, seed=0)
    for row in fx["y"]:
        for v in row:
            assert math.isfinite(v), f"non-finite value: {v}"


def test_layernorm_unit_gamma_beta_normalizes():
    """γ=1, β=0 なら、各行の出力 mean ≈ 0, var ≈ 1 (eps の影響でちょっとずれる)。"""
    rng = np.random.default_rng(42)
    x = rng.standard_normal((4, 16))
    gamma = np.ones(16)
    beta = np.zeros(16)
    y = _layernorm(x, gamma, beta, eps=1e-5)
    for r in range(4):
        m = y[r].mean()
        v = y[r].var()
        assert abs(m) < 1e-10, f"row {r}: mean={m}"
        # eps=1e-5 のせいで var はちょっと 1 より小さい (≈ var/(var+eps))
        assert abs(v - 1.0) < 0.01, f"row {r}: var={v}"


# ─── GELU ───────────────────────────────────────────────────────────────

def test_gelu_zero_at_zero():
    """GELU(0) = 0 (tanh(0)=0 より)。"""
    y = _gelu_tanh(np.array([0.0]))
    assert math.isclose(float(y[0]), 0.0, abs_tol=1e-15)


def test_gelu_positive_for_large_positive():
    """大きい正の入力では GELU(x) ≈ x になる (tanh が 1 に近づくので)。"""
    y = _gelu_tanh(np.array([5.0]))
    # 厳密値ではないが、おおよそ 5.0 に近いはず
    assert abs(float(y[0]) - 5.0) < 0.01


def test_gelu_near_zero_for_large_negative():
    """大きい負の入力では GELU(x) ≈ 0 になる (tanh が -1 に近いので)。"""
    y = _gelu_tanh(np.array([-5.0]))
    assert abs(float(y[0])) < 0.01


def test_gelu_finite_and_signed():
    fx = fix_gelu(32, seed=0)
    has_negative = any(v < 0 for v in fx["y"])
    has_positive = any(v > 0 for v in fx["y"])
    assert has_negative, "GELU は ReLU と違って負の出力もあるはず"
    assert has_positive
    for v in fx["y"]:
        assert math.isfinite(v)
