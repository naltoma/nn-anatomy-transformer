"""tools/inline_preset.py — デフォルト preset を src/js/presets.js に inline する。

設計書 §11 に対応。`presets/japanese-mini-v1.json` を読み込んで、ES module 形式の
`src/js/presets.js` を生成する。バンドル時にこのファイルが他の JS と一緒に
単一 HTML に展開されるので、起動時の fetch なしで preset が利用可能になる。

使い方:
    uv run python tools/inline_preset.py
    # → src/js/presets.js (上書き)

`make inline-preset` から呼ばれる。`make build-preset` で preset JSON を更新
した直後にこれを呼ぶこと。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_PRESET = ROOT / "presets" / "japanese-mini-v1.json"
OUT_JS = ROOT / "src" / "js" / "presets.js"

HEADER = """\
// presets.js — デフォルト preset (japanese-mini-v1) を inline 化したもの。
// 自動生成: tools/inline_preset.py が presets/japanese-mini-v1.json から書き出す。
// 直接編集禁止 — preset を変えるときは presets/ 側を更新してから
// `make inline-preset` を再実行すること。
//
// バンドル後の単一 HTML には起動時 fetch なしでこの preset が含まれる。
// 追加 preset (induction-head, knowledge-lookup, random-init 等) は
// 別ファイル (presets/*.json) のままで、UI から Import で読み込む想定。

"""


def main() -> int:
    obj = json.loads(SRC_PRESET.read_text(encoding="utf-8"))
    # JSON.stringify 相当を行うが、JS のコードに埋め込むので
    # 改行・インデントは詰めて書き出す (バンドルサイズを小さく抑えるため)。
    json_str = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    js_body = f"export const DEFAULT_PRESET = {json_str};\n"

    OUT_JS.parent.mkdir(parents=True, exist_ok=True)
    OUT_JS.write_text(HEADER + js_body, encoding="utf-8")

    size_kb = OUT_JS.stat().st_size / 1024
    print(f"✓ wrote {OUT_JS}  ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
