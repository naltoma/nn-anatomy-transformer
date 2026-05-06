"""数値参照実装 — JS の matrix.js が想定どおり計算しているか検証する基盤。

P1 段階では numpy で十分 (matmul / softmax / layerNorm / gelu はすべて式が単純)。
P3 以降で Self-Attention forward の参照を作るタイミングで PyTorch も導入し、
LayerNorm や Multi-Head Attention の挙動を完全に揃える。

設計方針:
- すべて float64 で計算 (JS の Float64Array と完全一致を狙う)。
- LayerNorm の var は分母 N (cols) の有偏分散、PyTorch の F.layer_norm と同じ。
- GELU は tanh 近似版を採用、JS 側 (matrix.js) もこの式に揃えてある。

JS 側のテスト (tests/js/test_matrix.test.mjs) は、ここで生成した fixture を
読み込んで、各演算の出力が 1e-12 オーダで一致するかを比較する。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES_DIR = ROOT / "tests" / "fixtures"


def _to_list(arr: np.ndarray) -> list:
    """ndarray を JSON シリアライズできる入れ子リストに。float64 で保存。"""
    return arr.astype(np.float64).tolist()


def _rng(seed: int) -> np.random.Generator:
    """同じ seed なら同じ乱数列。np.random.default_rng は PCG64 系で再現性 OK。"""
    return np.random.default_rng(seed)


# ─── matmul ────────────────────────────────────────────────────────────

def fix_matmul(M: int, K: int, N: int, seed: int) -> dict[str, Any]:
    """Random matmul fixture: A (M×K) · B (K×N) = C (M×N)."""
    rng = _rng(seed)
    A = rng.standard_normal((M, K), dtype=np.float64)
    B = rng.standard_normal((K, N), dtype=np.float64)
    C = A @ B
    return {
        "name": f"matmul_{M}x{K}_{K}x{N}_seed{seed}",
        "M": M, "K": K, "N": N,
        "A": _to_list(A),
        "B": _to_list(B),
        "C": _to_list(C),
    }


# ─── softmax ───────────────────────────────────────────────────────────

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数値安定化版の softmax (max を引いてから exp)。"""
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def fix_softmax(rows: int, cols: int, seed: int, scale: float = 1.0) -> dict[str, Any]:
    """Row-wise softmax (= 各行を独立に softmax)。"""
    rng = _rng(seed)
    x = rng.standard_normal((rows, cols), dtype=np.float64) * scale
    y = _softmax(x, axis=-1)
    return {
        "name": f"softmax_{rows}x{cols}_seed{seed}_scale{scale}",
        "rows": rows, "cols": cols, "scale": scale,
        "x": _to_list(x),
        "y": _to_list(y),
    }


# ─── LayerNorm ─────────────────────────────────────────────────────────

def _layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float) -> np.ndarray:
    """各行 (= 最終軸) ごとに正規化。PyTorch の F.layer_norm と同じ式:
        mean = sum(x) / cols  (バイアス補正なし)
        var  = sum((x-mean)^2) / cols
        y = (x - mean) / sqrt(var + eps) * gamma + beta
    """
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)  # numpy の var は ddof=0 = 有偏 (PyTorch と一致)
    inv_std = 1.0 / np.sqrt(var + eps)
    return (x - mean) * inv_std * gamma + beta


def fix_layernorm(rows: int, cols: int, seed: int) -> dict[str, Any]:
    """各行ごと LayerNorm。gamma, beta は ~1.0 / ~0.0 (preset 学習後を想定)。"""
    rng = _rng(seed)
    x = rng.standard_normal((rows, cols), dtype=np.float64)
    gamma = rng.standard_normal(cols, dtype=np.float64) * 0.1 + 1.0
    beta = rng.standard_normal(cols, dtype=np.float64) * 0.1
    eps = 1e-5
    y = _layernorm(x, gamma, beta, eps)
    return {
        "name": f"layernorm_{rows}x{cols}_seed{seed}",
        "rows": rows, "cols": cols, "eps": eps,
        "x": _to_list(x),
        "gamma": _to_list(gamma),
        "beta": _to_list(beta),
        "y": _to_list(y),
    }


# ─── GELU (tanh 近似) ───────────────────────────────────────────────────

_GELU_COEF = math.sqrt(2.0 / math.pi)


def _gelu_tanh(x: np.ndarray) -> np.ndarray:
    """GELU tanh 近似 (PyTorch F.gelu(x, approximate='tanh') と同式)。
        gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    """
    inner = _GELU_COEF * (x + 0.044715 * x ** 3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def fix_gelu(n: int, seed: int) -> dict[str, Any]:
    """GELU (tanh 近似)。matrix.js も同じ式を使うので、係数まで完全一致を期待する。"""
    rng = _rng(seed)
    x = rng.standard_normal(n, dtype=np.float64) * 2  # 広めの範囲
    y = _gelu_tanh(x)
    return {
        "name": f"gelu_{n}_seed{seed}",
        "n": n,
        "x": _to_list(x),
        "y": _to_list(y),
    }


# ─── 集約 ──────────────────────────────────────────────────────────────

def collect_fixtures() -> list[dict[str, Any]]:
    """全フィクスチャを生成。サイズと seed を散らして網羅性を上げる。"""
    out: list[dict[str, Any]] = []

    # matmul: 設計書の T=5, d_model=16 周辺をカバー
    for (M, K, N, s) in [
        (4, 5, 6, 0),
        (5, 16, 16, 1),    # T × d_model · d_model × d_k
        (16, 16, 16, 2),
        (1, 8, 8, 3),
        (3, 4, 7, 4),
        (5, 5, 16, 5),     # attention output (T × T) · (T × d_k)
    ]:
        out.append({"kind": "matmul", **fix_matmul(M, K, N, s)})

    # softmax: 行 (= 各 query) 単位の正規化、attention map 形状をカバー
    for (r, c, s, scale) in [
        (5, 5, 0, 1.0),    # T × T attention
        (5, 16, 1, 1.0),
        (3, 8, 2, 5.0),    # 大きい score 値 (発散しやすい状況の数値安定化テスト)
        (1, 16, 3, 0.1),
    ]:
        out.append({"kind": "softmax", **fix_softmax(r, c, s, scale)})

    # layernorm: 残差 + LN の出力形状 (T × d_model)
    for (r, c, s) in [(5, 16, 0), (1, 16, 1), (8, 16, 2), (5, 32, 3)]:
        out.append({"kind": "layernorm", **fix_layernorm(r, c, s)})

    # gelu: FFN 中間層 (T × d_ff)
    for (n, s) in [(16, 0), (32, 1), (80, 2)]:  # 80 = 5 × 16, FFN 入力相当
        out.append({"kind": "gelu", **fix_gelu(n, s)})

    return out


def write_fixtures(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fixtures = collect_fixtures()
    out_path.write_text(json.dumps(fixtures, ensure_ascii=False, indent=2))
    print(f"wrote {out_path} ({len(fixtures)} fixtures)")
