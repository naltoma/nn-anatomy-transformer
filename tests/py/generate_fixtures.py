"""Fixture 生成のエントリポイント。`make fixtures` から呼ばれる。

出力:
- tests/fixtures/p1_matrix.json : matrix.js の数値検証 (P1)
- tests/fixtures/p3_attention.json : Tier 1 Self-Attention forward の検証 (P3)。
  preset (japanese-mini-v1) を読み込んで 8 例文を numpy で forward した結果。

これらの fixture は JS 側から読み込んで「JS forward と numpy forward が
1e-12 オーダで一致する」ことを確認するために使う。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TOOLS = ROOT / "tools"

# このスクリプトを直接実行しても pytest 経由でも import できるよう、
# 関係するディレクトリを sys.path に足す。
for d in (HERE, TOOLS):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from reference import write_fixtures, FIXTURES_DIR  # noqa: E402
import build_preset as bp  # noqa: E402


def write_attention_fixtures(out_path: Path) -> None:
    """preset (japanese-mini-v1) を使って 8 例文を numpy で forward。
    JS 側で同じ計算をして数値一致するかを確認するための fixture。

    Multi-Head 対応で、Q/K/V/attnOut は (T, h, d_k)、scores/attn は (h, T, T) を
    JS の Float64Array フラット配列と同じ row-major で 1D に ravel して保存する。
    JS 側の net.Q[(t*h + hi) * d_k + k] と numpy (T,h,d_k).ravel() は同レイアウト。
    """
    net = bp.build_preset()
    fixtures = []
    for sample in bp.SAMPLE_SENTENCES_WITH_INTENT:
        token_ids = np.array(
            [bp.WORD2ID[w] for w in sample["tokens"]], dtype=np.int64
        )
        out = bp.forward_attention(net, token_ids)
        T = out["X"].shape[0]
        D = out["X"].shape[1]
        h = out["Q"].shape[1]
        d_k = out["Q"].shape[2]
        d_ff = out["ffn_h"].shape[1]
        fixtures.append({
            "name": sample["text"],
            "tokens": [int(i) for i in token_ids],
            "shapes": {
                "X": [T, D],
                "Q": [T, h, d_k],
                "K": [T, h, d_k],
                "V": [T, h, d_k],
                "scores": [h, T, T],
                "attn": [h, T, T],
                "attnOut": [T, h, d_k],
                "Y": [T, D],
                "residual1": [T, D],
                "ln1_out": [T, D],
                "ffn_pre": [T, d_ff],
                "ffn_h": [T, d_ff],
                "ffn_out": [T, D],
                "residual2": [T, D],
                "ln2_out": [T, D],
            },
            "X": out["X"].ravel().tolist(),
            "Q": out["Q"].ravel().tolist(),
            "K": out["K"].ravel().tolist(),
            "V": out["V"].ravel().tolist(),
            "scores": out["scores"].ravel().tolist(),
            "attn": out["attn"].ravel().tolist(),
            "attnOut": out["attnOut"].ravel().tolist(),
            "Y": out["Y"].ravel().tolist(),
            "residual1": out["residual1"].ravel().tolist(),
            "ln1_out": out["ln1_out"].ravel().tolist(),
            "ffn_pre": out["ffn_pre"].ravel().tolist(),
            "ffn_h": out["ffn_h"].ravel().tolist(),
            "ffn_out": out["ffn_out"].ravel().tolist(),
            "residual2": out["residual2"].ravel().tolist(),
            "ln2_out": out["ln2_out"].ravel().tolist(),
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fixtures, ensure_ascii=False, indent=2))
    print(f"wrote {out_path} ({len(fixtures)} attention fixtures)")


def main() -> int:
    write_fixtures(FIXTURES_DIR / "p1_matrix.json")
    write_attention_fixtures(FIXTURES_DIR / "p3_attention.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
