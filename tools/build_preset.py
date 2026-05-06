"""tools/build_preset.py — Hand-crafted Transformer preset の構築。

設計書 §6.9 の指針に従って、16 vocab × 16 dim の embedding と
W_Q / W_K / W_V / W_O を numpy で直接書き込み、JSON 出力する。
学習はしない。preset の意図は designIntent フィールドに記録する。

使い方:
    uv run python tools/build_preset.py
    # → presets/japanese-mini-v1.json

セルフチェック:
    生成と同時に 8 例文を numpy で forward して、意図した attention 関係
    (例: 「美しい→花」「これ→花」) が softmax 後 0.5 以上で出るかを確認する。
    閾値を満たさない場合は SystemExit して「設計の意図が壊れた」ことを警告。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PRESETS_DIR = ROOT / "presets"

# ─── ハイパパラメータ (設計書 §3 / §5 と同期) ─────────────────────

T = 5
D_MODEL = 16
H = 2             # Tier 2: 2 head
D_K = D_MODEL // H  # d_k = 8 (per-head)
D_FF = 32         # Tier 2 で使う (FFN は P7c)
N_BLOCKS = 1
VOCAB_SIZE = 16

VOCAB = [
    "これ",   #  0
    "それ",   #  1
    "私",     #  2
    "猫",     #  3
    "花",     #  4
    "本",     #  5
    "小説",   #  6
    "大きい", #  7
    "美しい", #  8
    "人気",   #  9
    "読む",   # 10
    "好き",   # 11
    "は",     # 12
    "が",     # 13
    "を",     # 14
    "です",   # 15
]
WORD2ID = {w: i for i, w in enumerate(VOCAB)}

# ─── Embedding 設計値 ─────────────────────────────────────────
# 設計書 §6.9.2 の表に従う。
# 1 次カテゴリ (is_noun, is_adj 等) は 3.0 で書き込み、PE のノイズ (max ±1) を
# 上回るようにする。2 次特徴 (美しさ, サイズ等) は 1.0、識別痕跡は 0.3。

CAT_VAL = 3.0     # 1 次カテゴリ強度
FEAT_VAL = 1.0    # 2 次意味特徴
TRACE_VAL = 0.3   # 識別痕跡

# Embedding 各次元の意味 (designIntent にも入れる)
EMBEDDING_DIM_INTENT = {
    0: "is_noun (猫, 花, 本, 小説 で {})".format(CAT_VAL),
    1: "is_adjective (大きい, 美しい, 人気 で {})".format(CAT_VAL),
    2: "is_pronoun (これ, それ, 私 で {})".format(CAT_VAL),
    3: "is_predicate (読む, 好き で {})".format(CAT_VAL),
    4: "is_は (は で {})".format(CAT_VAL),
    5: "is_が (が で {})".format(CAT_VAL),
    6: "is_を (を で {})".format(CAT_VAL),
    7: "is_です (です で {})".format(CAT_VAL),
    8: "美しさ特徴 (美しい, 花 で {})".format(FEAT_VAL),
    9: "サイズ特徴 (大きい, 猫 で {})".format(FEAT_VAL),
    10: "書物特徴 (本, 小説, 読む で {})".format(FEAT_VAL),
    11: "人気度特徴 (人気, 小説 で {})".format(FEAT_VAL),
    12: "識別痕跡 A (token id mod 4 == 0 のときに +{})".format(TRACE_VAL),
    13: "識別痕跡 B (token id mod 4 == 1)",
    14: "識別痕跡 C (token id mod 4 == 2)",
    15: "識別痕跡 D (token id mod 4 == 3)",
}


def build_W_E() -> np.ndarray:
    """設計書 §6.9.2 の表に従って Embedding を構築する。"""
    W = np.zeros((VOCAB_SIZE, D_MODEL), dtype=np.float64)

    # dim 0: is_noun (猫=3, 花=4, 本=5, 小説=6)
    for w in (3, 4, 5, 6):
        W[w, 0] = CAT_VAL
    # dim 1: is_adjective (大きい=7, 美しい=8, 人気=9)
    for w in (7, 8, 9):
        W[w, 1] = CAT_VAL
    # dim 2: is_pronoun (これ=0, それ=1, 私=2)
    for w in (0, 1, 2):
        W[w, 2] = CAT_VAL
    # dim 3: is_predicate (読む=10, 好き=11)
    for w in (10, 11):
        W[w, 3] = CAT_VAL
    # dim 4-7: 各助詞 / コピュラ
    W[12, 4] = CAT_VAL  # は
    W[13, 5] = CAT_VAL  # が
    W[14, 6] = CAT_VAL  # を
    W[15, 7] = CAT_VAL  # です
    # dim 8: 美しさ特徴 (美しい=8, 花=4)
    W[8, 8] = FEAT_VAL
    W[4, 8] = FEAT_VAL
    # dim 9: サイズ特徴 (大きい=7, 猫=3)
    W[7, 9] = FEAT_VAL
    W[3, 9] = FEAT_VAL
    # dim 10: 書物特徴 (本=5, 小説=6, 読む=10)
    W[5, 10] = FEAT_VAL
    W[6, 10] = FEAT_VAL
    W[10, 10] = FEAT_VAL
    # dim 11: 人気度特徴 (人気=9, 小説=6)
    W[9, 11] = FEAT_VAL
    W[6, 11] = FEAT_VAL
    # dim 12-15: 緩い識別痕跡 (token を区別するための微小な差)
    for i in range(VOCAB_SIZE):
        W[i, 12 + (i % 4)] += TRACE_VAL

    return W


def build_W_P() -> np.ndarray:
    """位置エンコーディング (Transformer 元論文方式)。
        PE(t, 2i)   = sin(t / 10000^(2i / d_model))
        PE(t, 2i+1) = cos(t / 10000^(2i / d_model))
    """
    P = np.zeros((T, D_MODEL), dtype=np.float64)
    for t in range(T):
        for i in range(D_MODEL // 2):
            theta = t / (10000 ** (2 * i / D_MODEL))
            P[t, 2 * i] = math.sin(theta)
            P[t, 2 * i + 1] = math.cos(theta)
    return P


# ─── W_Q / W_K の bond dim 設計 (Tier 2: h=2 multi-head) ─────
# 設計書 §6.9.3 + Multi-Head 拡張。bond 0/1/2 を 2 head に分散する:
#   head 0 (出力 col 0..7):
#     bond 0 (head0 col 0 = global col 0): adjective Q → noun K
#     bond 1 (head0 col 1 = global col 1): pronoun  Q → noun K
#   head 1 (出力 col 8..15):
#     bond 2 (head1 col 0 = global col 8): predicate Q → noun K
# bond 用に使わない col は 0 のまま。
# multi-head の意義は「並列に異なる関係を見られる」こと。head 0 が修飾語族、
# head 1 が述語族を担当する設計。

ATTN_W = 1.0  # W_Q[input_dim, bond_col] のスケール

# 各 bond の設計 (Tier 2)
# bond_col は「グローバル」col index (= 0..d_model-1)。
# head_idx はその col がどの head に属するかを示す (head_idx = bond_col // d_k)。
ATTENTION_BONDS = {
    0: {
        "intent": "adjective Q → noun K (head 0)",
        "Q_input_dim": 1,   # is_adjective
        "K_input_dim": 0,   # is_noun
        "bond_col": 0,      # head 0 の col 0
    },
    1: {
        "intent": "pronoun Q → noun K (head 0)",
        "Q_input_dim": 2,   # is_pronoun
        "K_input_dim": 0,   # is_noun
        "bond_col": 1,      # head 0 の col 1
    },
    2: {
        "intent": "predicate Q → noun K (head 1)",
        "Q_input_dim": 3,   # is_predicate
        "K_input_dim": 0,   # is_noun
        "bond_col": 8,      # head 1 の col 0 (= global col 8)
    },
}


def build_W_Q() -> np.ndarray:
    """各 bond ごとに「query input dim → bond_col」の射影を立てる。
    shape は (d_model, h * d_k) = (16, 16)。h>1 のときは cols が head 別に分かれる。"""
    W = np.zeros((D_MODEL, H * D_K), dtype=np.float64)
    for bond in ATTENTION_BONDS.values():
        W[bond["Q_input_dim"], bond["bond_col"]] = ATTN_W
    return W


def build_W_K() -> np.ndarray:
    """各 bond ごとに「key input dim → bond_col」の射影を立てる。"""
    W = np.zeros((D_MODEL, H * D_K), dtype=np.float64)
    for bond in ATTENTION_BONDS.values():
        W[bond["K_input_dim"], bond["bond_col"]] = ATTN_W
    return W


def build_W_V() -> np.ndarray:
    """V = X · W_V。Tier 2 でも shape は (d_model, h*d_k) = (16, 16) で
    identity を使う。こうすると各 head は X の同じ位置の値をそのまま V として渡し、
    attention output は「注目した位置の X」を集約したものになり解釈しやすい。
    head 0 の V 出力は X[:, 0..7]、head 1 の V 出力は X[:, 8..15] になる。"""
    return np.eye(D_MODEL, dtype=np.float64)


def build_W_O() -> np.ndarray:
    """Y = concat(head_outputs) · W_O。shape (h*d_k, d_model) = (16, 16)。
    Tier 2 では identity でスタートする (concat 後そのまま出力)。
    本格的には W_O に学習で意味を持たせるが、本シミュレータでは hand-crafted の単位行列で
    「concat したベクトルがそのまま Y になる」状態を維持する。"""
    return np.eye(D_MODEL, dtype=np.float64)


# ─── LayerNorm パラメータ (Tier 2: 残差 + LN) ────────────────
# LN(x) = γ * (x - μ) / sqrt(σ² + ε) + β  (per-token、d_model 軸方向の正規化)
# 教育用に γ = 1, β = 0 で固定。これで「正規化だけ」が観察できる (学習要素ゼロ)。

LN_EPS = 1e-5  # PyTorch の F.layer_norm デフォルト


def build_LN_gamma() -> np.ndarray:
    """LayerNorm の γ (スケール)。本 preset は全 dim で +1.000。"""
    return np.ones(D_MODEL, dtype=np.float64)


def build_LN_beta() -> np.ndarray:
    """LayerNorm の β (シフト)。本 preset は全 dim で 0.000。"""
    return np.zeros(D_MODEL, dtype=np.float64)


# ─── FFN パラメータ (Tier 2: 2 層 MLP, GELU) ────────────────
# FFN は per-token に作用する。入力 LN1_out (T, d_model=16) を
#   H = GELU(LN1_out · W1 + b1)   shape (T, d_ff=32)
#   FFN_out = H · W2 + b2          shape (T, d_model=16)
# として変換。
# preset 設計の意図 (§6.9.4): 特定の dim パターンに反応する「検出器ニューロン」を作り、
# それを W2 で対応する出力 dim に書き戻すことで、attention 後の token 表現に
# 「カテゴリ + 特徴」の組み合わせを強化する knowledge memory として動かす。

# 検出器のスレッショルド (LN 後の値が ~+1.5 以上の dim を「立っている」と判定)
DET_W = 1.0       # 単 dim 検出器の重み
DET_B = -1.0      # 単 dim 検出器のバイアス (= GELU 入力 = ln1_out[d] - 1.0)
COMB_W = 0.6      # 組み合わせ検出器の重み
COMB_B = -0.5     # 組み合わせ検出器のバイアス
ENHANCE = 0.5     # AND 検出器が発火したときの feature 強化量


def build_FFN_W1() -> np.ndarray:
    """W1: shape (d_model=16, d_ff=32)。32 個の検出器ニューロン。
    h0..h7: 1 次カテゴリ + 助詞 dim 単独検出器 (d0..d7 → h0..h7)
    h8..h11: 2 次特徴 dim 単独検出器 (d8..d11 → h8..h11)
    h12..h15: AND 検出器 (名詞 ∧ 各特徴)
    h16..h31: 未使用 (= 0)
    """
    W = np.zeros((D_MODEL, D_FF), dtype=np.float64)
    # h0..h11: 単 dim 検出器
    for hk in range(12):
        W[hk, hk] = DET_W
    # h12..h15: AND 検出器 (名詞 dim 0 + 特徴 dim 8..11)
    for k, feat_dim in enumerate([8, 9, 10, 11]):
        hk = 12 + k
        W[0, hk] = COMB_W       # 名詞 dim 入力
        W[feat_dim, hk] = COMB_W  # 特徴 dim 入力
    # h16..h31: 未使用 (W = 0)
    return W


def build_FFN_b1() -> np.ndarray:
    """b1: shape (d_ff=32,)。検出器バイアス。"""
    b = np.zeros(D_FF, dtype=np.float64)
    for hk in range(12):
        b[hk] = DET_B
    for hk in range(12, 16):
        b[hk] = COMB_B
    # h16..h31: 0 (= 出力も常に 0、GELU 後も 0)
    return b


def build_FFN_W2() -> np.ndarray:
    """W2: shape (d_ff=32, d_model=16)。検出器の活性を出力 dim に書き戻す。
    h0..h11: identity (検出器の発火を対応 dim にそのまま戻す → feature 維持)
    h12 (名詞 ∧ 美しさ) → 出力 d8 に +ENHANCE (美しさ特徴を強化)
    h13 (名詞 ∧ サイズ) → 出力 d9
    h14 (名詞 ∧ 書物)   → 出力 d10
    h15 (名詞 ∧ 人気度) → 出力 d11
    h16..h31: 何も書き戻さない (= 0)
    """
    W = np.zeros((D_FF, D_MODEL), dtype=np.float64)
    # h0..h11: identity passthrough
    for hk in range(12):
        W[hk, hk] = 1.0
    # h12..h15: AND 検出器 → 対応する特徴 dim に書き戻し (強化)
    for k, feat_dim in enumerate([8, 9, 10, 11]):
        hk = 12 + k
        W[hk, feat_dim] = ENHANCE
    return W


def build_FFN_b2() -> np.ndarray:
    """b2: shape (d_model=16,)。本 preset は全 0。"""
    return np.zeros(D_MODEL, dtype=np.float64)


# 検出器の意図 (designIntent に出すラベル)
FFN_NEURON_INTENT = {
    0: "名詞 dim (d0) 検出",
    1: "形容詞 dim (d1) 検出",
    2: "指示詞 dim (d2) 検出",
    3: "述語 dim (d3) 検出",
    4: "助詞「は」(d4) 検出",
    5: "助詞「が」(d5) 検出",
    6: "助詞「を」(d6) 検出",
    7: "コピュラ「です」(d7) 検出",
    8: "美しさ特徴 (d8) 検出",
    9: "サイズ特徴 (d9) 検出",
    10: "書物特徴 (d10) 検出",
    11: "人気度特徴 (d11) 検出",
    12: "AND: 名詞 ∧ 美しさ → 美しい名詞検出 (d8 を強化)",
    13: "AND: 名詞 ∧ サイズ → サイズを持つ名詞検出 (d9 を強化)",
    14: "AND: 名詞 ∧ 書物 → 書物検出 (d10 を強化)",
    15: "AND: 名詞 ∧ 人気度 → 人気な名詞検出 (d11 を強化)",
}


# ─── Forward (numpy) ─────────────────────────────────────────
# self-check 専用の素朴な forward。Tier 1 だけサポート。

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def _layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = LN_EPS) -> np.ndarray:
    """LayerNorm を per-token (= 行ごとに d_model 軸で) 計算。
    PyTorch F.layer_norm 互換 (有偏分散、分母は N=d_model)。"""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)  # numpy の var は ddof=0 (有偏分散)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta


# tanh 近似版 GELU (元論文 / matrix.js の gelu と同一)
_GELU_COEF = math.sqrt(2.0 / math.pi)

def _gelu(x: np.ndarray) -> np.ndarray:
    """tanh 近似版 GELU。matrix.js の gelu と完全一致。"""
    inner = _GELU_COEF * (x + 0.044715 * x ** 3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def forward_attention(net: dict, tokens: np.ndarray) -> dict:
    """Self-Attention forward を numpy で計算 (Tier 2: Multi-Head + 残差 + LN)。
    内部の各テンソル (X, Q, K, V, scores, attn, attnOut, Y, residual1, ln1_out) を返す。

    Multi-Head 構造 (Lesson 8 まで):
      Q_full = X @ W_Q  shape (T, h*d_k)
      Q_full を (T, h, d_k) にリシェイプ → 各 head h で
        scores_h = (Q_h · K_h^T) / √d_k  shape (T, T)
        attn_h   = softmax(scores_h)    shape (T, T)
        out_h    = attn_h · V_h         shape (T, d_k)
      out を head 軸で concat → (T, h*d_k) → · W_O → Y (T, d_model)

    残差 + LN (Lesson 9):
      residual1 = X + Y                              # (T, d_model)
      ln1_out   = LayerNorm(residual1, γ, β)         # (T, d_model)

    返り値の各キーは (T, h, ...) の形に整形して JS 側と揃える。
      Q, K, V    : (T, h, d_k)
      scores, attn: (h, T, T)
      attnOut    : (T, h, d_k)
      Y          : (T, d_model)
      residual1  : (T, d_model)
      ln1_out    : (T, d_model)
    """
    W_E = net["W_E"]
    W_P = net["W_P"]
    W_Q = net["W_Q"]
    W_K = net["W_K"]
    W_V = net["W_V"]
    W_O = net["W_O"]
    LN_gamma = net["LN_gamma"]
    LN_beta = net["LN_beta"]

    h = H
    d_k = D_K
    Tn = tokens.shape[0]

    # Embedding + PE
    X = W_E[tokens] + W_P                 # (T, d_model)
    # Linear projections (concat across heads)
    Q_full = X @ W_Q                       # (T, h*d_k)
    K_full = X @ W_K                       # (T, h*d_k)
    V_full = X @ W_V                       # (T, h*d_k)

    # Reshape to (T, h, d_k)
    Q_h = Q_full.reshape(Tn, h, d_k)
    K_h = K_full.reshape(Tn, h, d_k)
    V_h = V_full.reshape(Tn, h, d_k)

    inv_sqrt_dk = 1.0 / math.sqrt(d_k)
    scores_h = np.zeros((h, Tn, Tn), dtype=np.float64)
    attn_h   = np.zeros((h, Tn, Tn), dtype=np.float64)
    out_h    = np.zeros((Tn, h, d_k), dtype=np.float64)
    for hi in range(h):
        Qh = Q_h[:, hi, :]   # (T, d_k)
        Kh = K_h[:, hi, :]   # (T, d_k)
        Vh = V_h[:, hi, :]   # (T, d_k)
        s = (Qh @ Kh.T) * inv_sqrt_dk
        a = _softmax(s, axis=-1)
        scores_h[hi] = s
        attn_h[hi] = a
        out_h[:, hi, :] = a @ Vh

    # concat & W_O
    attnOut_concat = out_h.reshape(Tn, h * d_k)   # (T, h*d_k)
    Y = attnOut_concat @ W_O                       # (T, d_model)

    # 残差接続 + LayerNorm (Lesson 9)
    residual1 = X + Y                              # (T, d_model)
    ln1_out = _layer_norm(residual1, LN_gamma, LN_beta)

    # FFN (Lesson 10): 2 層 MLP + GELU
    W1 = net["FFN_W1"]; b1 = net["FFN_b1"]
    W2 = net["FFN_W2"]; b2 = net["FFN_b2"]
    ffn_pre = ln1_out @ W1 + b1                    # (T, d_ff) — GELU 入力
    ffn_h = _gelu(ffn_pre)                          # (T, d_ff) — GELU 出力
    ffn_out = ffn_h @ W2 + b2                       # (T, d_model)

    # 2 回目の残差 + LayerNorm
    residual2 = ln1_out + ffn_out                   # (T, d_model)
    ln2_out = _layer_norm(residual2, LN_gamma, LN_beta)  # 同じ LN_gamma/beta を流用

    return {
        "X": X,
        "Q": Q_h, "K": K_h, "V": V_h,
        "scores": scores_h, "attn": attn_h,
        "attnOut": out_h, "Y": Y,
        "residual1": residual1,
        "ln1_out": ln1_out,
        "ffn_pre": ffn_pre,
        "ffn_h": ffn_h,
        "ffn_out": ffn_out,
        "residual2": residual2,
        "ln2_out": ln2_out,
    }


# ─── 例文と意図した attention 関係 ───────────────────────────

# 設計書 §6.2 と整合。各文ごとに「query 位置 → 期待 key 位置」の主要関係を列挙。
SAMPLE_SENTENCES_WITH_INTENT = [
    {
        "text": "これ は 美しい 花 です",
        "tokens": ["これ", "は", "美しい", "花", "です"],
        "expected": [
            # (description, query_pos, key_pos, min_attn)
            ("これ → 花 (pron→noun)",   0, 3, 0.5),
            ("美しい → 花 (adj→noun)",   2, 3, 0.5),
        ],
    },
    {
        "text": "それ は 人気 小説 です",
        "tokens": ["それ", "は", "人気", "小説", "です"],
        "expected": [
            ("それ → 小説 (pron→noun)",  0, 3, 0.5),
            ("人気 → 小説 (adj→noun)",   2, 3, 0.5),
        ],
    },
    {
        "text": "私 は 猫 が 好き",
        "tokens": ["私", "は", "猫", "が", "好き"],
        "expected": [
            ("私 → 猫 (pron→noun)",      0, 2, 0.5),
            ("好き → 猫 (pred→noun)",    4, 2, 0.5),
        ],
    },
    {
        "text": "私 は 本 を 読む",
        "tokens": ["私", "は", "本", "を", "読む"],
        "expected": [
            ("私 → 本 (pron→noun)",      0, 2, 0.5),
            ("読む → 本 (pred→noun)",    4, 2, 0.5),
        ],
    },
    {
        "text": "私 は 小説 を 読む",
        "tokens": ["私", "は", "小説", "を", "読む"],
        "expected": [
            ("私 → 小説 (pron→noun)",    0, 2, 0.5),
            ("読む → 小説 (pred→noun)",  4, 2, 0.5),
        ],
    },
    {
        "text": "大きい 猫 は 美しい です",
        "tokens": ["大きい", "猫", "は", "美しい", "です"],
        "expected": [
            ("大きい → 猫 (adj→noun)",   0, 1, 0.5),
            ("美しい → 猫 (adj→noun)",   3, 1, 0.5),
        ],
    },
    {
        "text": "美しい 花 は 人気 です",
        "tokens": ["美しい", "花", "は", "人気", "です"],
        "expected": [
            ("美しい → 花 (adj→noun)",   0, 1, 0.5),
            ("人気 → 花 (adj→noun)",     3, 1, 0.5),
        ],
    },
    {
        "text": "これ は 大きい 本 です",
        "tokens": ["これ", "は", "大きい", "本", "です"],
        "expected": [
            ("これ → 本 (pron→noun)",    0, 3, 0.5),
            ("大きい → 本 (adj→noun)",   2, 3, 0.5),
        ],
    },
]


def self_check(net: dict) -> int:
    """8 例文を forward し、意図した attention 関係の attention 値が
    閾値以上で出るかを head 軸方向の最大値で確認する (multi-head 対応)。
    失敗が 1 件でもあれば 1 を返す。"""
    failures = 0
    print("\n── Self-check: 意図した attention pattern が head 軸の最大で出るか ──")
    for sample in SAMPLE_SENTENCES_WITH_INTENT:
        token_ids = np.array([WORD2ID[w] for w in sample["tokens"]], dtype=np.int64)
        out = forward_attention(net, token_ids)
        attn = out["attn"]  # (h, T, T)
        for desc, q_pos, k_pos, min_attn in sample["expected"]:
            # 各 head の attn 値を取って、最大の head が閾値以上を満たすか
            per_head = attn[:, q_pos, k_pos]
            best_h = int(np.argmax(per_head))
            val = float(per_head[best_h])
            mark = "✓" if val >= min_attn else "✗"
            print(f"  [{mark}] {sample['text']}  {desc}: head{best_h} = {val:.3f}  (>= {min_attn})")
            if val < min_attn:
                failures += 1
    print(f"── self-check 結果: {failures} 件失敗 ──\n")
    return failures


# ─── JSON 出力 ───────────────────────────────────────────────

def to_nested(arr: np.ndarray) -> list:
    """ndarray を JSON シリアライズ可能な入れ子リストに。float64 で保存。"""
    return arr.astype(np.float64).tolist()


def build_preset() -> dict:
    """全パラメータを構築して dict として返す。"""
    return {
        "W_E": build_W_E(),
        "W_P": build_W_P(),
        "W_Q": build_W_Q(),
        "W_K": build_W_K(),
        "W_V": build_W_V(),
        "W_O": build_W_O(),
        "LN_gamma": build_LN_gamma(),
        "LN_beta": build_LN_beta(),
        "FFN_W1": build_FFN_W1(),
        "FFN_b1": build_FFN_b1(),
        "FFN_W2": build_FFN_W2(),
        "FFN_b2": build_FFN_b2(),
    }


def to_json_obj(net: dict) -> dict:
    """設計書 §6.9.5 の JSON フォーマットに合わせて整形する。"""
    return {
        "name": "japanese-mini-v1",
        "version": 2,
        "kind": "hand-crafted",
        "tier": 2,
        "description": (
            "Hand-crafted Tier 2 preset (Multi-Head Self-Attention + 残差 + LayerNorm + FFN,"
            " h=2, d_k=8, d_ff=32)."
            " 設計書 §6.9 の指針に従って構築 (学習はしていない)。"
            " head 0 = adj→noun + pron→noun (修飾語族),"
            " head 1 = pred→noun (述語族),"
            " LN は γ=1, β=0 で標準化のみ,"
            " FFN は単 dim 検出器 (h0..h11) と AND 検出器 (h12..h15: 名詞 ∧ 各特徴) の knowledge memory 設計"
            " を意図的に作り込んである。"
        ),
        "config": {
            "T": T,
            "d_model": D_MODEL,
            "h": H,
            "d_k": D_K,
            "d_ff": D_FF,
            "N": N_BLOCKS,
            "vocabSize": VOCAB_SIZE,
            "vocab": list(VOCAB),
        },
        "weights": {
            "W_E": to_nested(net["W_E"]),
            "W_P": to_nested(net["W_P"]),
            "blocks": [
                {
                    "W_Q": to_nested(net["W_Q"]),
                    "W_K": to_nested(net["W_K"]),
                    "W_V": to_nested(net["W_V"]),
                    "W_O": to_nested(net["W_O"]),
                    # 残差 + LN1: γ=1, β=0 (本シミュレータは LN を「標準化のみ」に固定)
                    "LN1_gamma": to_nested(net["LN_gamma"]),
                    "LN1_beta":  to_nested(net["LN_beta"]),
                    # FFN: 2 層 MLP + GELU (検出器ニューロン設計)
                    "FFN_W1": to_nested(net["FFN_W1"]),
                    "FFN_b1": to_nested(net["FFN_b1"]),
                    "FFN_W2": to_nested(net["FFN_W2"]),
                    "FFN_b2": to_nested(net["FFN_b2"]),
                    # FFN 後の 2 回目残差 + LN2 (同じ γ=1, β=0 を流用)
                    "LN2_gamma": to_nested(net["LN_gamma"]),
                    "LN2_beta":  to_nested(net["LN_beta"]),
                },
            ],
        },
        "designIntent": {
            "embeddingDims": EMBEDDING_DIM_INTENT,
            "attentionBonds": {
                str(k): v["intent"] for k, v in ATTENTION_BONDS.items()
            },
            "ffnNeurons": FFN_NEURON_INTENT,
            "scales": {
                "category_value": CAT_VAL,
                "feature_value": FEAT_VAL,
                "trace_value": TRACE_VAL,
                "attention_W": ATTN_W,
                "sqrt_d_k": math.sqrt(D_K),
            },
        },
    }


def main() -> int:
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PRESETS_DIR / "japanese-mini-v1.json"

    net = build_preset()
    failures = self_check(net)
    if failures > 0:
        print(f"❌ Self-check で {failures} 件失敗。preset の設計を見直してください。",
              file=sys.stderr)
        return 1

    obj = to_json_obj(net)
    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    size_kb = out_path.stat().st_size / 1024
    print(f"✓ wrote {out_path}  ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
