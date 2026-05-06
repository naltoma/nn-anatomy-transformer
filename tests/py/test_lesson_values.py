"""Lesson 1〜5 の hint / checks に書いた具体的な数値が、preset を改変しても
壊れていないか自動検証する (Tier 2: h=2, d_k=8 対応)。tolerance は表記桁 (3 桁)
より厳しい 5e-4。

bond 配置 (Tier 2):
  bond 0 (head 0 col 0 = global col 0): adj  (Q) → noun (K)
  bond 1 (head 0 col 1 = global col 1): pron (Q) → noun (K)
  bond 2 (head 1 col 0 = global col 8): pred (Q) → noun (K)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TOOLS = ROOT / "tools"
for d in (HERE, TOOLS):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

import build_preset as bp  # noqa: E402


# ─── ヘルパ ───────────────────────────────────────────────

TOL = 5e-4


def _net_and_forward(sample_words: list[str]) -> tuple[dict, dict]:
    """preset と forward 結果を返す。各 lesson テストで使い回す。"""
    net = bp.build_preset()
    tokens = np.array([bp.WORD2ID[w] for w in sample_words], dtype=np.int64)
    res = bp.forward_attention(net, tokens)
    return net, res


# ─── Lesson 1: token 埋め込みと X の合成 ──────────────────

def test_lesson1_token_embedding_categorical_dims():
    """Lesson 1 hint: dim 0..3 = カテゴリ (1 次, +3.0)、dim 8..11 = 特徴 (2 次, +1.0)。"""
    net = bp.build_preset()
    W_E = net["W_E"]
    # 1 次カテゴリ +3.0
    assert abs(W_E[bp.WORD2ID["美しい"], 1] - 3.0) < TOL  # is_adjective
    assert abs(W_E[bp.WORD2ID["花"], 0] - 3.0) < TOL       # is_noun
    assert abs(W_E[bp.WORD2ID["これ"], 2] - 3.0) < TOL     # is_pronoun
    assert abs(W_E[bp.WORD2ID["好き"], 3] - 3.0) < TOL     # is_predicate

    # 2 次特徴 +1.0
    assert abs(W_E[bp.WORD2ID["美しい"], 8] - 1.0) < TOL   # 美しさ特徴
    assert abs(W_E[bp.WORD2ID["花"], 8] - 1.0) < TOL        # 美しさ特徴 (花も持つ)
    assert abs(W_E[bp.WORD2ID["猫"], 9] - 1.0) < TOL        # サイズ特徴

    # 助詞は全カテゴリ 0
    for d in range(4):
        assert abs(W_E[bp.WORD2ID["は"], d]) < TOL
    # 「です」も全カテゴリ 0 (Lesson 5 の sample #1 で T4 です が flat になる根拠)
    for d in range(4):
        assert abs(W_E[bp.WORD2ID["です"], d]) < TOL


def test_lesson1_X_with_PE_for_T2_美しい():
    """Lesson 1 task 1: X[T2 美しい, d1] が +3.0 付近 (PE で少し変動)。"""
    _, res = _net_and_forward(["これ", "は", "美しい", "花", "です"])
    X = res["X"]
    # X[T2, d1] = W_E[美しい, d1] (+3.0) + W_P[t=2, d1]
    # W_P[t=2, d1] = cos(2 / 10000^(0/16)) = cos(2) ≈ -0.4161
    expected = 3.0 + math.cos(2.0)
    assert abs(X[2, 1] - expected) < TOL
    # ほぼ +2.584
    assert abs(X[2, 1] - 2.5839) < 1e-3


# ─── Lesson 2: 位置エンコーディング ──────────────────────

def test_lesson2_W_P_pos0_is_zero_one_pattern():
    """Lesson 2 task 1: pos 0 は d0=sin(0)=0, d1=cos(0)=1, ..."""
    net = bp.build_preset()
    W_P = net["W_P"]
    assert abs(W_P[0, 0] - 0.0) < TOL
    assert abs(W_P[0, 1] - 1.0) < TOL
    assert abs(W_P[0, 2] - 0.0) < TOL
    assert abs(W_P[0, 3] - 1.0) < TOL


def test_lesson2_W_P_independent_of_token():
    """Lesson 2 task 2: W_P は token と無関係 (sample 切替で変わらない)。"""
    net1 = bp.build_preset()
    net2 = bp.build_preset()  # 何度作っても同じ
    np.testing.assert_array_equal(net1["W_P"], net2["W_P"])


def test_lesson2_W_P_high_dim_nearly_constant_across_pos():
    """Lesson 2 task 3: 高 dim (例: d14, d15) は 10000^(14/16) ≈ 5623 が分母なので
    pos 0..4 でほぼ一定値。"""
    net = bp.build_preset()
    W_P = net["W_P"]
    # d14 (sin), d15 (cos) は pos 0..4 でほぼ同じ値 (差 < 2e-3)
    # 比較として d0 (sin, 周期 2π) は spread が 1.6 程度になる。
    for d in (14, 15):
        col = W_P[:, d]
        spread = col.max() - col.min()
        assert spread < 2e-3, f"d{d} の pos 軸方向の広がり {spread} が大きい"
    # 低 dim (d0) は spread 大きい (> 1.5)
    assert (W_P[:, 0].max() - W_P[:, 0].min()) > 1.5


# ─── Lesson 3: Q/K/V projection (bond 配線) ───────────────

def test_lesson3_W_Q_bond_wiring():
    """Lesson 3 task 1: W_Q の非ゼロ位置は 3 つだけ。
    bond 0 (head 0 col 0): W_Q[d1, d0] = 1 (adj→noun)
    bond 1 (head 0 col 1): W_Q[d2, d1] = 1 (pron→noun)
    bond 2 (head 1 col 0 = global col 8): W_Q[d3, d8] = 1 (pred→noun)"""
    net = bp.build_preset()
    W_Q = net["W_Q"]
    assert abs(W_Q[1, 0] - 1.0) < TOL  # bond 0
    assert abs(W_Q[2, 1] - 1.0) < TOL  # bond 1
    assert abs(W_Q[3, 8] - 1.0) < TOL  # bond 2 (head 1)
    nonzero = np.argwhere(np.abs(W_Q) > TOL)
    nonzero_set = {tuple(p) for p in nonzero}
    assert nonzero_set == {(1, 0), (2, 1), (3, 8)}, \
        f"W_Q の非ゼロ位置が想定外: {nonzero_set}"


def test_lesson3_Q_T2_美しい_profile():
    """Lesson 3 task 2/3: Q[T2 美しい] を head 別に観察。
    head 0 col 0 = +2.584 (本物の adj→bond 0)、
    head 0 col 1 = +0.591 (PE 漏れ → bond 1 col)、
    head 1 col 0 = +0.807 (PE 漏れ → bond 2 col)。"""
    _, res = _net_and_forward(["これ", "は", "美しい", "花", "です"])
    Q = res["Q"]  # shape (T, h, d_k)
    assert abs(Q[2, 0, 0] - 2.584) < 1e-3
    assert abs(Q[2, 0, 1] - 0.591) < 1e-3
    for k in range(2, 8):
        assert abs(Q[2, 0, k]) < TOL
    assert abs(Q[2, 1, 0] - 0.807) < 1e-3
    for k in range(1, 8):
        assert abs(Q[2, 1, k]) < TOL


# ─── Lesson 4: K (Key) projection ────────────────────────

def test_lesson4_W_K_noun_routes_to_three_bonds():
    """Lesson 4 task 1: W_K の非ゼロ位置は名詞 dim → 3 bond の 3 個だけ。
    W_K[d0, d0]=1, W_K[d0, d1]=1, W_K[d0, d8]=1。"""
    net = bp.build_preset()
    W_K = net["W_K"]
    assert abs(W_K[0, 0] - 1.0) < TOL
    assert abs(W_K[0, 1] - 1.0) < TOL
    assert abs(W_K[0, 8] - 1.0) < TOL
    nonzero = np.argwhere(np.abs(W_K) > TOL)
    nonzero_set = {tuple(p) for p in nonzero}
    assert nonzero_set == {(0, 0), (0, 1), (0, 8)}


def test_lesson4_K_T3_花_lights_all_three_bonds():
    """Lesson 4 task 2/3: K[T3 花] (名詞) は 3 つの bond すべてで +3.141。
    head 0 col 0 = head 0 col 1 = head 1 col 0 = +3.141 (= is_noun + sin(3))。"""
    _, res = _net_and_forward(["これ", "は", "美しい", "花", "です"])
    K = res["K"]  # (T, h, d_k)
    assert abs(K[3, 0, 0] - 3.141) < 1e-3   # head 0, bond 0
    assert abs(K[3, 0, 1] - 3.141) < 1e-3   # head 0, bond 1
    assert abs(K[3, 1, 0] - 3.141) < 1e-3   # head 1, bond 2
    # 他の col は 0
    for k in range(2, 8):
        assert abs(K[3, 0, k]) < TOL
    for k in range(1, 8):
        assert abs(K[3, 1, k]) < TOL


def test_lesson4_K_T0_これ_all_zero():
    """Lesson 4 task 3 補足: K[T0 これ] (t=0 かつ非名詞) は head 0/1 ともすべて 0。
    X[T0, d0] = is_noun(0) + sin(0) = 0 で、PE 漏れ経路も 0 になる。"""
    _, res = _net_and_forward(["これ", "は", "美しい", "花", "です"])
    K = res["K"]  # (T, h, d_k)
    for hi in range(K.shape[1]):
        for k in range(K.shape[2]):
            assert abs(K[0, hi, k]) < TOL


# ─── Lesson 5: V (Value) projection ──────────────────────

def test_lesson5_V_equals_X():
    """Lesson 5 task 1: W_V = identity なので V_full = X。
    head 0 → X[:, 0..7]、head 1 → X[:, 8..15] と分割される。"""
    _, res = _net_and_forward(["これ", "は", "美しい", "花", "です"])
    V = res["V"]; X = res["X"]
    T, h, d_k = V.shape
    for t in range(T):
        # head 0 と X[:, 0..7] の一致
        for k in range(d_k):
            assert abs(V[t, 0, k] - X[t, k]) < TOL
        # head 1 と X[:, 8..15] の一致
        for k in range(d_k):
            assert abs(V[t, 1, k] - X[t, d_k + k]) < TOL


def test_lesson5_V_T3_花_features_in_head1():
    """Lesson 5 task 3: V[T3 花, head 1, d0] = 美しさ特徴 +1.0 + 微小 PE = +1.030。
    特徴系 dim (X の d8..d15) が head 1 に乗る。"""
    _, res = _net_and_forward(["これ", "は", "美しい", "花", "です"])
    V = res["V"]
    # head 1 d0 = X[T3, d8] = 美しさ特徴 +1 + W_P[3, 8] ≈ +0.030
    assert abs(V[3, 1, 0] - 1.030) < 1e-3
    # head 1 d4 = X[T3, d12] = 識別痕跡 +0.3 (花 の id = 4 mod 4 = 0) + 微小 PE ≈ +0.003
    assert abs(V[3, 1, 4] - 0.303) < 1e-3


# ─── Lesson 4: scores (Q · K^T / √d_k) ──────────────────

def test_lesson6_scores_sample1():
    """Lesson 6: sample #1 「これ は 美しい 花 です」の scores 代表値。
    head 0 (modifier→noun): T0/T2 → T3 花 が大きく出る。
    head 1 (predicate→noun): sample #1 には predicate token が無いので、
      T0 row の peak は T3 = +1.111 程度 (PE 漏れ × is_noun)。"""
    _, res = _net_and_forward(["これ", "は", "美しい", "花", "です"])
    scores = res["scores"]  # (h, T, T)
    # head 0 task 1: scores[T2 美しい, T3 花] = +3.526 (bond 0 + PE 漏れ)
    assert abs(scores[0, 2, 3] - 3.526) < 1e-3
    # head 0 task 2: scores[T0 これ, T3 花] = +4.442 (bond 1 + PE 漏れ)
    assert abs(scores[0, 0, 3] - 4.442) < 1e-3
    # head 0 task 3 補足: scores[T1 は, T3 花] = +0.945 (PE 漏れだけ)
    assert abs(scores[0, 1, 3] - 0.945) < 1e-3
    # head 0: 名詞行 T3 花 は本当に小さい (Q[T3] が PE 漏れだけ)
    assert scores[0, 3].max() < 0.1   # max は T4 の +0.047
    # head 1: sample #1 には predicate なし。T0 row の peak は T3 = +1.111
    assert abs(scores[1, 0, 3] - 1.111) < 1e-3
    # T0 列 (= K[T0 これ]) は head 0/1 とも全行 0
    # (K[T0, *] の bond cols は X[T0, d0]=0 + sin(0)=0 で 0、つまり Q · K[T0] = 0)
    for hi in range(scores.shape[0]):
        for i in range(scores.shape[1]):
            assert abs(scores[hi, i, 0]) < TOL


def test_lesson6_scores_breakdown_at_head0_T2_T3():
    """Lesson 6 task 1: scores head 0 [T2, T3] の内訳。
    head 0 col 0: Q[T2, h0, c0]=+2.584 × K[T3, h0, c0]=+3.141 = +8.116 (bond 0 本筋)
    head 0 col 1: Q[T2, h0, c1]=+0.591 × K[T3, h0, c1]=+3.141 = +1.857 (PE 漏れ)
    総和 +9.973 / √8 = +3.526。"""
    _, res = _net_and_forward(["これ", "は", "美しい", "花", "です"])
    Q = res["Q"]; K = res["K"]
    qk = Q[2, 0, :] * K[3, 0, :]  # head 0
    assert abs(qk[0] - 8.116) < 1e-3
    assert abs(qk[1] - 1.857) < 1e-3
    # head 0 col 2..7 は 0
    for k in range(2, 8):
        assert abs(qk[k]) < TOL
    total = qk.sum()
    assert abs(total - 9.973) < 1e-3
    assert abs(total / math.sqrt(8) - 3.526) < 1e-3


# ─── Lesson 5: softmax と attention map ─────────────────

def test_lesson7_attn_sample1():
    """Lesson 7: sample #1 の attn 代表値。head 0 (modifier→noun)。"""
    _, res = _net_and_forward(["これ", "は", "美しい", "花", "です"])
    attn = res["attn"]  # (h, T, T)
    # head 0 task 1: attn[T0 これ, T3 花] = 0.912 (bond 1 集中)
    assert abs(attn[0, 0, 3] - 0.912) < 1e-3
    # head 0 task 2: attn[T2 美しい, T3 花] = 0.834 (bond 0 集中)
    assert abs(attn[0, 2, 3] - 0.834) < 1e-3
    # head 0 task 3: T3 花 (名詞) 行は flat (max 0.221)
    assert attn[0, 3].max() < 0.23
    assert attn[0, 3].min() > 0.17
    # 各 head の各行は softmax なので和が 1
    for hi in range(attn.shape[0]):
        np.testing.assert_array_almost_equal(attn[hi].sum(axis=1), np.ones(5))


def test_lesson7_attn_sample3_bond2_in_head1():
    """Lesson 7 task 3 補足: sample #3 で bond 2 (pred→noun) は head 1 に立つ。
    T4「好き」→T2「猫」 attn[head 1] = 0.948、
    T0「私」→T2「猫」 attn[head 0] = 0.977 (head 0 の bond 1 で立つ)。"""
    _, res = _net_and_forward(["私", "は", "猫", "が", "好き"])
    attn = res["attn"]  # (h, T, T)
    # head 0: T0 私 → T2 猫 (bond 1)
    assert abs(attn[0, 0, 2] - 0.977) < 1e-3
    # head 1: T4 好き → T2 猫 (bond 2)
    assert abs(attn[1, 4, 2] - 0.948) < 1e-3


# ─── Lesson 8: Multi-Head 役割分担 ──────────────────────

def test_lesson8_attnOut_head1_carries_only_size_increment():
    """Lesson 8 task 3 (head 1): T0「私」の attnOut [head 1] は「猫」のサイズ特徴 (d1 = +1)
    を baseline +1.0 に対する +0.489 の上乗せとして受け取る。
    具体値: attnOut[T0, h1, d1] ≈ +1.489 = 他 token の baseline +1.0
            + 「猫」のサイズ特徴 +1 × attn(T0→T2) = 0.489"""
    _, res = _net_and_forward(["私", "は", "猫", "が", "好き"])
    attnOut = res["attnOut"]
    V = res["V"]
    attn = res["attn"]
    # head 1 attn[T0=私, T2=猫] が約 0.489
    assert abs(attn[1, 0, 2] - 0.489) < 1e-3
    # attnOut[T0, h1, d0] が +0.019 付近
    assert abs(attnOut[0, 1, 0] - 0.019) < 1e-3
    # attnOut[T0, h1, d1] が +1.489 付近 (size 特徴の +0.489 上乗せ)
    assert abs(attnOut[0, 1, 1] - 1.489) < 1e-3
    # attnOut[T0, h1, d2] が +0.006 付近
    assert abs(attnOut[0, 1, 2] - 0.006) < 1e-3
    # V[T2 猫, h1, d1] が +2.000 (size feature +1 + cos PE +1)
    assert abs(V[2, 1, 1] - 2.000) < 1e-3


def test_lesson8_attnOut_T4_好き_copies_V_T2_猫_in_head1():
    """Lesson 8 task 3: T4「好き」 + head 1 で attnOut が V[T2「猫」, h1] のほぼコピー。
    attn[h1, T4「好き」, T2「猫」] = 0.948 と独占的なので、サイズ特徴 (d1 = +2.000) と
    識別痕跡 D (d7 = +1.300) が「好き」の attnOut [h1] に丸ごと届く。
    一方 head 0 では attn[h0, T4, T2] = 0.273 と分散していて、attnOut [h0] は薄い受信。"""
    _, res = _net_and_forward(["私", "は", "猫", "が", "好き"])
    attnOut = res["attnOut"]
    V = res["V"]
    attn = res["attn"]
    # head 1 attn[T4=好き, T2=猫] が約 0.948
    assert abs(attn[1, 4, 2] - 0.948) < 1e-3
    # attnOut[T4, h1] と V[T2 猫, h1] の各 dim の差が小さい (各 dim < 0.06)
    diff_h1 = np.abs(attnOut[4, 1, :] - V[2, 1, :]).max()
    assert diff_h1 < 0.06, f"head 1 で V のコピー差が大きい: {diff_h1}"
    # 特に d1 が +1.948 付近 (V[T2, h1, d1] = +2.000 の 0.948 倍 + 他 baseline)
    assert abs(attnOut[4, 1, 1] - 1.948) < 5e-3
    # head 0 では attn が分散 (0.273 が最大) で V[T2 猫, h0] のコピーにはならない
    assert abs(attn[0, 4, 2] - 0.273) < 1e-3
    diff_h0 = np.abs(attnOut[4, 0, :] - V[2, 0, :]).max()
    assert diff_h0 > 0.5, f"head 0 では V[T2] と乖離するはずだが {diff_h0}"


def test_lesson8_attnOut_T0_私_copies_V_T2_猫_in_head0():
    """Lesson 8 task 3: sample #3 で T0「私」の attnOut [head 0] が
    V[T2「猫」, head 0] のほぼコピーになる。
    attn[head 0, T0, T2] ≈ 0.977 で T2 がほぼ独占的なため。
    例: attnOut[T0, h0, d0] = +3.831 ≈ 0.977 × V[T2, h0, d0] (+3.91) + 微小寄与"""
    _, res = _net_and_forward(["私", "は", "猫", "が", "好き"])
    attnOut = res["attnOut"]   # (T, h, d_k)
    V = res["V"]
    # attnOut[T0, head 0, d0] が dominant な +3.831 付近
    assert abs(attnOut[0, 0, 0] - 3.831) < 5e-3
    # V[T2 猫, head 0, d0] は +3.91 付近
    assert abs(V[2, 0, 0] - 3.91) < 5e-2
    # head 0 では attnOut[T0] と V[T2] の差が小さい (各 dim で < 0.1)
    diff = np.abs(attnOut[0, 0, :] - V[2, 0, :]).max()
    assert diff < 0.15
    # head 1 では attn が 0.489 程度なので V[T2] からのずれは大きい
    diff_h1 = np.abs(attnOut[0, 1, :] - V[2, 1, :]).max()
    assert diff_h1 > 0.3


# ─── Lesson 9: 残差接続 + LayerNorm ──────────────────────

def test_lesson9_residual_preserves_pronoun_dim_sample3():
    """Lesson 9 task 1: T0「私」d2 (指示詞 dim) は X = +3.000 が
    残差で保持される。Y は +0.598 (位置エンコーディングの混入) なので、
    residual1 = +3.000 + +0.598 = +3.598。"""
    _, res = _net_and_forward(["私", "は", "猫", "が", "好き"])
    X = res["X"]; Y = res["Y"]; residual1 = res["residual1"]
    # T0 私, d2 (= 指示詞 dim)
    assert abs(X[0, 2] - 3.000) < 1e-3
    assert abs(Y[0, 2] - 0.598) < 1e-3
    assert abs(residual1[0, 2] - 3.598) < 1e-3
    # T0 私, d0 (= 名詞 dim): X = 0、Y は V[T2 猫] のコピーで +3.831
    assert abs(X[0, 0]) < 1e-9
    assert abs(Y[0, 0] - 3.831) < 5e-3
    assert abs(residual1[0, 0] - 3.831) < 5e-3


def test_lesson9_layernorm_normalizes_each_token_row():
    """Lesson 9 task 2/3: ln1_out の各行は mean=0, std≈1 に標準化される。
    残差で生じた token 間のスケール差が消える。"""
    _, res = _net_and_forward(["私", "は", "猫", "が", "好き"])
    ln1_out = res["ln1_out"]
    for t in range(ln1_out.shape[0]):
        row = ln1_out[t]
        assert abs(row.mean()) < 1e-5, f"T{t} mean = {row.mean()}"
        assert abs(row.std() - 1.0) < 1e-3, f"T{t} std = {row.std()}"


def test_lesson10_ffn_neurons_fire_for_T0_私():
    """Lesson 10 task 1: T0「私」 で h0 (名詞), h2 (指示詞), h13 (名詞∧サイズ),
    h15 (名詞∧人気度) が発火する。"""
    _, res = _net_and_forward(["私", "は", "猫", "が", "好き"])
    ffn_h = res["ffn_h"]  # (T, d_ff)
    # T0「私」の各検出器
    assert ffn_h[0, 0] > 0.5, f"h0 (名詞) should fire: got {ffn_h[0, 0]}"
    assert ffn_h[0, 2] > 0.5, f"h2 (指示詞) should fire: got {ffn_h[0, 2]}"
    assert ffn_h[0, 13] > 0.5, f"h13 (名詞∧サイズ) should fire"
    assert ffn_h[0, 15] > 0.5, f"h15 (名詞∧人気度) should fire"
    # 発火しない検出器
    assert ffn_h[0, 3] < 0.1, f"h3 (述語) should not fire"
    assert ffn_h[0, 4] < 0.1, f"h4 (は dim) should not fire"


def test_lesson10_ffn_neurons_fire_for_each_function_token():
    """Lesson 10 task 2: 各 token がその文法カテゴリに対応する単 dim 検出器を発火させる。
    T1「は」→ h4、T3「が」→ h5、T4「好き」→ h3。"""
    _, res = _net_and_forward(["私", "は", "猫", "が", "好き"])
    ffn_h = res["ffn_h"]
    assert ffn_h[1, 4] > 0.5, f"T1「は」 → h4: got {ffn_h[1, 4]}"
    assert ffn_h[3, 5] > 0.5, f"T3「が」 → h5: got {ffn_h[3, 5]}"
    assert ffn_h[4, 3] > 0.5, f"T4「好き」 → h3: got {ffn_h[4, 3]}"


def test_lesson10_block_output_ln2_normalized():
    """Lesson 10 task 3: ln2_out (block 最終出力) の各行が mean=0, std≈1 に正規化される。"""
    _, res = _net_and_forward(["私", "は", "猫", "が", "好き"])
    ln2_out = res["ln2_out"]
    for t in range(ln2_out.shape[0]):
        row = ln2_out[t]
        assert abs(row.mean()) < 1e-5, f"T{t} mean = {row.mean()}"
        assert abs(row.std() - 1.0) < 1e-3, f"T{t} std = {row.std()}"


def test_lesson9_layernorm_T0_私_d0_value():
    """Lesson 9 task 2 の具体例: T0「私」の ln1_out[d0] = +1.913 付近。
    residual1[T0, :] の mean = +1.453、std = 1.243 で、
    (residual1[T0, d0] - mean) / std = (+3.831 - 1.453) / 1.243 ≈ +1.913"""
    _, res = _net_and_forward(["私", "は", "猫", "が", "好き"])
    ln1_out = res["ln1_out"]
    residual1 = res["residual1"]
    # 行平均と標準偏差の確認
    row = residual1[0]
    assert abs(row.mean() - 1.453) < 1e-3
    assert abs(row.std() - 1.243) < 1e-3
    # ln1_out の値
    assert abs(ln1_out[0, 0] - 1.913) < 5e-3
    assert abs(ln1_out[0, 2] - 1.726) < 5e-3


def test_lesson8_multihead_role_separation_sample3():
    """Lesson 8 task 1〜2: sample #3 で head 0 / head 1 が異なる関係を同時並行で拾う。
    head 0 の T0「私」→T2「猫」 = 0.977 (bond 1, 鋭い集中)
    head 0 の T4「好き」→T2「猫」 = 0.273 (bond 2 は head 0 担当外、PE 漏れだけで弱い)
    head 1 の T4「好き」→T2「猫」 = 0.948 (bond 2, 鋭い集中)
    head 1 の T0「私」→T2「猫」 = 0.489 (PE 漏れによる緩い集中)"""
    _, res = _net_and_forward(["私", "は", "猫", "が", "好き"])
    attn = res["attn"]  # (h, T, T)
    # head 0 では bond 1 (pron→noun) が鋭く、bond 2 は弱い
    assert abs(attn[0, 0, 2] - 0.977) < 1e-3
    assert abs(attn[0, 4, 2] - 0.273) < 1e-3
    # head 1 では bond 2 (pred→noun) が鋭く、bond 1 はぼやける
    assert abs(attn[1, 4, 2] - 0.948) < 1e-3
    assert abs(attn[1, 0, 2] - 0.489) < 1e-3
    # 各 head の各行は softmax なので和が 1
    for hi in range(attn.shape[0]):
        np.testing.assert_array_almost_equal(attn[hi].sum(axis=1), np.ones(5))
