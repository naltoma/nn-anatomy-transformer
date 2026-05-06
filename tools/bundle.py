#!/usr/bin/env python3
"""tools/bundle.py — src/ を build/nn_sim.html に単一 HTML として展開する。

設計書 §11, §15.4, §10 に対応。

役割:
- src/index.html の `<link rel="stylesheet" href="./style.css">` を
  `<style>` 直書きに置き換える。
- `<script type="module" src="./js/controller.js">` を、src/js/ 配下の
  全モジュールを依存順に連結した単一の `<script type="module">` に置き換える。
- ES モジュール間の `import ... from "./foo.js";` はすべて削除し、
  `export const/function ...` からは `export ` を剥がす。全モジュールが
  同一スコープに展開されるので、元の名前で互いを参照できる。

制約:
- 外部依存は一切持たない (Python 標準ライブラリのみ)。
- 追加の `uv add` は行わない (設計書 §15.4 より)。
- CDN 参照・外部フォントなども作らない (設計書 §10 「単一 HTML」)。

使い方:
    uv run python tools/bundle.py
    # → build/nn_sim.html (単一 HTML)

Makefile から `make bundle` で呼ばれることを想定。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
BUILD = ROOT / "build"
OUT = BUILD / "nn_sim_transformer.html"

# src/js 配下を連結する順序。末端 (import されるだけの側) を先に置く。
#   rng.js        : 依存なし (PE 生成と内部の決定的乱数のみ)
#   tokenizer.js  : 依存なし (16 語固定語彙 + encode/decode)
#   matrix.js     : 依存なし (flat-array 行列演算ヘルパ)
#   presets.js    : 依存なし (デフォルト preset = japanese-mini-v1 を inline)
#   lessons.js    : 依存なし (Lesson T1-T9 定義)
#   model.js      : rng.js + matrix.js を使う (Transformer の forward のみ)
#   view.js       : matrix.js を使う (行列ヒートマップ + pan/zoom)
#   explain.js    : state を受け取って「式の展開」パネルの HTML を返すだけ
#   controller.js : 上記全部に依存。エントリポイント。
MODULE_ORDER = (
    "rng.js",
    "tokenizer.js",
    "matrix.js",
    "presets.js",
    "lessons.js",
    "model.js",
    "view.js",
    "explain.js",
    "controller.js",
)

# 単行 or 複数行の import 文を行ごと丸ごと食う。
# "[^;]*" は Python では既定で改行を含むので、複数行にまたがる import も OK。
IMPORT_RE = re.compile(r"^import\s[^;]*;[ \t]*\n?", re.MULTILINE)
# 先頭の `export ` だけを剥がす (`export const/function/class` すべてに対応)。
EXPORT_RE = re.compile(r"^export\s+", re.MULTILINE)

# サイズ予算 (設計書 §10)。超えたら警告だけ出して止めない。
# Transformer 版は MLP 版より行列計算と多 Block コードで増えるが、
# Backward / 学習を削ったことで MLP 版 (175 KB) + 行列ヒートマップ追加分で
# 250 KB を上限とする。デフォルト preset (~20-30 KB) を inline する分も含む。
SIZE_BUDGET_BYTES = 250 * 1024


def strip_module_syntax(js: str) -> str:
    """import 文を削除し、`export ` キーワードを剥がす。

    これだけで、連結後の単一 <script type="module"> 内で全モジュールの
    トップレベル名 (関数・定数) が共有されるようになる。
    """
    js = IMPORT_RE.sub("", js)
    js = EXPORT_RE.sub("", js)
    return js


# トップレベル宣言 (function / const / let / var) の名前を拾う。
# 行頭 ^ に空白・`export ` が来るケースをまとめて面倒みる。
TOP_LEVEL_DECL_RE = re.compile(
    r"^(?:export\s+)?(?:function\s+(\w+)|(?:const|let|var)\s+(\w+))\b",
    re.MULTILINE,
)


def detect_toplevel_collisions(sources: dict[str, str]) -> list[str]:
    """複数モジュールで同じ top-level 名が宣言されていないかを確認する。

    ES モジュールは各ファイルが個別スコープだが、本バンドラは単一
    <script type="module"> に結合するため、同名の `function` / `const` が
    ファイルをまたいで現れると `SyntaxError: has already been declared`
    で確実に壊れる。`make bundle` の時点で早めに気付くための検査。
    """
    seen: dict[str, list[str]] = {}
    for fname, body in sources.items():
        for m in TOP_LEVEL_DECL_RE.finditer(body):
            name = m.group(1) or m.group(2)
            seen.setdefault(name, []).append(fname)
    errors: list[str] = []
    for name, files in seen.items():
        if len(files) > 1:
            # 同一ファイル内での重複は普通ないが、一応集計して誤報を避ける。
            uniq = sorted(set(files))
            if len(uniq) > 1:
                errors.append(
                    f"top-level name collision: `{name}` in {uniq}"
                )
    return errors


def ensure_no_html_tokens(name: str, body: str) -> None:
    """`</script>` / `</style>` が JS/CSS 側に紛れ込んでいないか確認する。

    もし含まれているとインライン化したときにタグが途中で閉じてしまうため、
    ビルドを中断する。現行のソースには存在しないはずなので、回帰防止用。
    """
    lowered = body.lower()
    for bad in ("</script", "</style"):
        if bad in lowered:
            raise SystemExit(
                f"ERROR: {name} に `{bad}` が含まれている。"
                f"インライン化すると HTML が壊れるので中断。"
            )


def inline_stylesheet(html: str, style_css: str) -> str:
    """`<link rel="stylesheet" href="./style.css">` を `<style>…</style>` に。"""
    link_re = re.compile(
        r'<link\s+[^>]*rel="stylesheet"[^>]*href="\./style\.css"[^>]*/?\s*>'
    )
    if link_re.search(html) is None:
        raise SystemExit(
            'ERROR: src/index.html に `<link rel="stylesheet" href="./style.css">` が無い'
        )
    # re.sub の replacement は \1 等のバックリファレンスを解釈するため、
    # 安全に差し込むには lambda を使って「生の文字列」を返させる。
    replacement = f"<style>\n{style_css}\n</style>"
    return link_re.sub(lambda _m: replacement, html, count=1)


def inline_scripts(html: str, bundled_js: str) -> str:
    """`<script type="module" src="./js/controller.js"></script>` を
    全モジュール連結した `<script type="module">…</script>` に。
    """
    script_re = re.compile(
        r'<script\s+type="module"\s+src="\./js/controller\.js"\s*>\s*</script>'
    )
    if script_re.search(html) is None:
        raise SystemExit(
            'ERROR: src/index.html に '
            '`<script type="module" src="./js/controller.js"></script>` が無い'
        )
    replacement = f'<script type="module">\n{bundled_js}\n</script>'
    return script_re.sub(lambda _m: replacement, html, count=1)


def build_bundle() -> str:
    """src/js/*.js を依存順に読み、連結済み JS 文字列を返す。"""
    # まず全ソースを読んで衝突検査。衝突があれば bundle せず中断する。
    sources: dict[str, str] = {}
    for name in MODULE_ORDER:
        path = SRC / "js" / name
        raw = path.read_text(encoding="utf-8")
        ensure_no_html_tokens(name, raw)
        sources[name] = raw
    collisions = detect_toplevel_collisions(sources)
    if collisions:
        msg = "ERROR: 単一 <script> にまとめると壊れる名前衝突が見つかった:\n" + \
              "\n".join(f"  - {c}" for c in collisions) + \
              "\n  → どちらかの名前を変えて import/export 側もそろえてください。"
        raise SystemExit(msg)

    parts: list[str] = [
        "// === bundled by tools/bundle.py — 直接編集しないこと ===",
        "// src/js/*.js を依存順に連結し、import 文を削除し、`export ` を剥がした。",
        "// 詳細: tools/bundle.py のモジュール docstring を参照。",
    ]
    for name in MODULE_ORDER:
        stripped = strip_module_syntax(sources[name])
        parts.append(f"\n// ---------- {name} ----------\n")
        parts.append(stripped.rstrip() + "\n")
    return "\n".join(parts)


def main(argv: list[str]) -> int:
    index_path = SRC / "index.html"
    style_path = SRC / "style.css"
    if not index_path.exists() or not style_path.exists():
        print(f"ERROR: {index_path} または {style_path} が無い", file=sys.stderr)
        return 1

    html = index_path.read_text(encoding="utf-8")
    style_css = style_path.read_text(encoding="utf-8")
    ensure_no_html_tokens("style.css", style_css)

    html = inline_stylesheet(html, style_css)
    bundled_js = build_bundle()
    html = inline_scripts(html, bundled_js)

    # `./style.css` や `./js/...js` への参照が他に残っていないかをチェック。
    # (将来 index.html に link/script が増えたときのガード)
    leftover = re.findall(r'href="\./style\.css"|src="\./js/[^"]+"', html)
    if leftover:
        print(
            f"ERROR: 未差し替えの外部参照が残っている: {leftover}",
            file=sys.stderr,
        )
        return 1

    BUILD.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")

    size = OUT.stat().st_size
    rel = OUT.relative_to(ROOT)
    msg = f"wrote {rel} ({size:,} bytes = {size / 1024:.1f} KB)"
    print(msg)
    if size > SIZE_BUDGET_BYTES:
        print(
            f"WARN: 設計書 §10 の 150 KB 目標を超えている ({size / 1024:.1f} KB)",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
