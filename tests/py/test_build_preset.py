"""build_preset.py で生成される preset の整合性チェック。

(1) JSON 構造が設計書 §6.9.5 のフォーマットを満たす
(2) Embedding が設計書 §6.9.2 の表どおりの値を持つ
(3) W_Q / W_K の bond が設計どおり (adj→noun, pron→noun, pred→noun)
(4) 8 例文で意図した attention 関係が softmax 後 0.5 以上で出る (= self-check)

build_preset.py 自体が main() で self-check を行うが、ここでは独立した test
として再確認することで、preset の壊れを CI で検出できるようにする。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

# tools/build_preset.py を import するためのパス調整
ROOT = Path(__file__).resolve().parent.parent.parent
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_preset as bp  # noqa: E402


# ─── Embedding (§6.9.2) ──────────────────────────────────────

def test_embedding_is_noun_dim():
    W_E = bp.build_W_E()
    # 名詞 (3, 4, 5, 6) は dim 0 で CAT_VAL
    for w in (3, 4, 5, 6):
        assert W_E[w, 0] == bp.CAT_VAL
    # 名詞以外は dim 0 がゼロ
    for w in range(bp.VOCAB_SIZE):
        if w not in (3, 4, 5, 6):
            assert W_E[w, 0] == 0.0


def test_embedding_is_adjective_dim():
    W_E = bp.build_W_E()
    for w in (7, 8, 9):
        assert W_E[w, 1] == bp.CAT_VAL
    for w in range(bp.VOCAB_SIZE):
        if w not in (7, 8, 9):
            assert W_E[w, 1] == 0.0


def test_embedding_is_pronoun_dim():
    W_E = bp.build_W_E()
    for w in (0, 1, 2):
        assert W_E[w, 2] == bp.CAT_VAL


def test_embedding_is_predicate_dim():
    W_E = bp.build_W_E()
    for w in (10, 11):
        assert W_E[w, 3] == bp.CAT_VAL


def test_embedding_particle_and_copula_dims():
    W_E = bp.build_W_E()
    assert W_E[12, 4] == bp.CAT_VAL  # は
    assert W_E[13, 5] == bp.CAT_VAL  # が
    assert W_E[14, 6] == bp.CAT_VAL  # を
    assert W_E[15, 7] == bp.CAT_VAL  # です


def test_embedding_semantic_features():
    W_E = bp.build_W_E()
    # 美しさ (dim 8): 美しい=8, 花=4
    assert W_E[8, 8] == bp.FEAT_VAL
    assert W_E[4, 8] == bp.FEAT_VAL
    # 書物 (dim 10): 本=5, 小説=6, 読む=10
    assert W_E[5, 10] == bp.FEAT_VAL
    assert W_E[6, 10] == bp.FEAT_VAL
    assert W_E[10, 10] == bp.FEAT_VAL


# ─── PE ──────────────────────────────────────────────────────

def test_positional_encoding_t0_is_alternating_zero_one():
    P = bp.build_W_P()
    # t=0 では sin(0) = 0, cos(0) = 1 になるので、各偶数次元 = 0、奇数次元 = 1
    for i in range(bp.D_MODEL // 2):
        assert P[0, 2 * i] == 0.0
        assert P[0, 2 * i + 1] == 1.0


def test_positional_encoding_shape():
    P = bp.build_W_P()
    assert P.shape == (bp.T, bp.D_MODEL)


# ─── W_Q / W_K (§6.9.3) ──────────────────────────────────────

def test_W_Q_attention_bonds_tier2():
    """Tier 2: bond 0/1 は head 0 (cols 0,1)、bond 2 は head 1 (col 8)。"""
    W_Q = bp.build_W_Q()
    assert W_Q[1, 0] == bp.ATTN_W   # bond 0 = adj→head 0 col 0
    assert W_Q[2, 1] == bp.ATTN_W   # bond 1 = pron→head 0 col 1
    assert W_Q[3, 8] == bp.ATTN_W   # bond 2 = pred→head 1 col 0 (= global col 8)
    nonzero = np.argwhere(W_Q != 0)
    expected = {(1, 0), (2, 1), (3, 8)}
    assert set(map(tuple, nonzero)) == expected


def test_W_K_attention_bonds_tier2():
    """Tier 2: noun が 3 つの bond すべて (cols 0, 1, 8) に流れる。"""
    W_K = bp.build_W_K()
    assert W_K[0, 0] == bp.ATTN_W
    assert W_K[0, 1] == bp.ATTN_W
    assert W_K[0, 8] == bp.ATTN_W   # head 1
    nonzero = np.argwhere(W_K != 0)
    expected = {(0, 0), (0, 1), (0, 8)}
    assert set(map(tuple, nonzero)) == expected


def test_W_V_W_O_are_identity():
    """V = X, Y = attnOut · W_O (Tier 2 でも W_V = W_O = identity を維持)。"""
    np.testing.assert_array_equal(bp.build_W_V(), np.eye(bp.D_MODEL))
    np.testing.assert_array_equal(bp.build_W_O(), np.eye(bp.D_MODEL))


# ─── Self-check (§6.9 全体) ──────────────────────────────────

def test_self_check_all_attention_patterns_satisfied():
    net = bp.build_preset()
    failures = bp.self_check(net)
    assert failures == 0, (
        f"{failures} 件の attention pattern が閾値 0.5 を下回った。"
        f"preset の設計を見直してください。"
    )


# ─── 個別の例文 (重要なものは独立 test に) ─────────────────────

@pytest.mark.parametrize("sample_idx", range(8))
def test_sample_sentence_attention(sample_idx):
    """各例文の主要 attention 関係が、いずれかの head で softmax 後 0.5 以上であること
    (Tier 2: head 軸方向の最大値で判定)。"""
    net = bp.build_preset()
    sample = bp.SAMPLE_SENTENCES_WITH_INTENT[sample_idx]
    token_ids = np.array([bp.WORD2ID[w] for w in sample["tokens"]], dtype=np.int64)
    out = bp.forward_attention(net, token_ids)
    attn = out["attn"]   # (h, T, T)
    for desc, q_pos, k_pos, min_attn in sample["expected"]:
        per_head = attn[:, q_pos, k_pos]
        val = float(per_head.max())
        assert val >= min_attn, (
            f"{sample['text']}: {desc}: max-head attn[{q_pos},{k_pos}]={val:.3f} < {min_attn}"
        )


# ─── JSON 構造 (§6.9.5) ──────────────────────────────────────

def test_to_json_obj_structure():
    """設計書 §6.9.5 で定義された JSON フォーマットを満たすこと。"""
    net = bp.build_preset()
    obj = bp.to_json_obj(net)
    # 必須トップレベルキー
    for key in ("name", "version", "kind", "tier", "description",
                "config", "weights", "designIntent"):
        assert key in obj, f"top-level key 欠落: {key}"
    # config の中身
    config = obj["config"]
    assert config["T"] == 5
    assert config["d_model"] == 16
    assert config["h"] == 2     # Tier 2: 2 head
    assert config["d_k"] == 8   # d_model / h
    assert config["vocabSize"] == 16
    assert len(config["vocab"]) == 16
    # weights の中身
    w = obj["weights"]
    assert "W_E" in w
    assert "W_P" in w
    assert "blocks" in w
    assert len(w["blocks"]) == 1
    block = w["blocks"][0]
    for key in ("W_Q", "W_K", "W_V", "W_O"):
        assert key in block, f"block にキー欠落: {key}"
    # kind = "hand-crafted"
    assert obj["kind"] == "hand-crafted"


def test_json_round_trip_size_within_budget():
    """生成された JSON を文字列化したサイズが 50 KB 以下であること
    (バンドル時に inline するので、preset は小さく抑えたい)。"""
    net = bp.build_preset()
    obj = bp.to_json_obj(net)
    s = json.dumps(obj, ensure_ascii=False)
    size_kb = len(s.encode("utf-8")) / 1024
    assert size_kb < 50, f"preset JSON が大きすぎる: {size_kb:.1f} KB"


def test_designIntent_lists_all_attention_bonds():
    """designIntent.attentionBonds に bond 0/1/2 の意図が記載されていること。"""
    net = bp.build_preset()
    obj = bp.to_json_obj(net)
    bonds = obj["designIntent"]["attentionBonds"]
    assert "0" in bonds and "adjective" in bonds["0"]
    assert "1" in bonds and "pronoun" in bonds["1"]
    assert "2" in bonds and "predicate" in bonds["2"]
