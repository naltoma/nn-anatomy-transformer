# 初期 README (アーカイブ)

> このファイルは v0.4 (2026-05) 以前の README.md を docs/ に移動したもの。
> プロジェクトが P0 (設計フェーズ) の段階で書かれた、実装計画寄りの紹介文。
> v1 リリース後は利用者向けの新 README.md がプロジェクト直下に置かれているのでそちらを参照。
> 本ファイルは履歴記録目的で残してある。

---

# NN Anatomy — Transformer (WIP)

Transformer の **inference (推論) 動作** を **行列のセル 1 個ずつ何が起きているか** の粒度で観察できるブラウザ用シミュレータ。姉妹プロジェクト [nn-anatomy (MLP 版)](https://github.com/naltoma/nn-anatomy) の哲学を Transformer に拡張したもので、大学情報系学部の **3〜4 年次以上** (深層学習・NLP の応用科目) を対象にする。

> **ステータス: 設計フェーズ (P0)**
> 設計書を [`design.md`](design.md) に書いた段階。実装はこれから (P1: matrix.js + tokenizer + PyTorch 参照実装)。
> 動くものはまだ無い。MLP 版が完成版なので、まずはそちらを参照のこと。

> **方針**: forward (推論) のみ。学習機能はシミュレータ本体に含めない。重みは別途 numpy で **hand-crafted (人手構築)** で書いた preset JSON を読み込んで使う。8 サンプル × T=5 で SGD 学習させても attention map が綺麗に分化しないリスクが高いため、教育的には人手で意図したパターンを埋め込むほうが目的に適う、という判断。Backward / 学習の核は MLP 版で習得済みであることが前提。

---

## 想定する完成像 (Tier 3)

- Transformer Block 1〜2 段、Multi-Head Self-Attention (h=2)、残差接続 + LayerNorm + FFN
- Self-Attention の Q/K/V 射影、Attention Score (Q·K^T)、softmax、value 重み付き和、を **すべて行列ヒートマップ** で表示
- Attention map の行・列ヘッダに **日本語の単語** が並ぶ (例: 「これ」→「花」: 0.65 のように直接読める)
- 各セルをクリックすると「式の展開」パネルにそのセルの計算を 3 段板書 ([段1] 一般形 → [段2] 当てはめ → [段3] 数値)
- 行列が画面に収まらないので **pan / zoom 必須** (SVG viewBox 操作で実装)
- 入力文は **16 語固定の日本語語彙** から組まれる短文 (T=5 トークン、例: 「これ は 美しい 花 です」)
- 重みは **hand-crafted preset の JSON** を読み込み、attention map が言語的に意味を持つ状態を保証 (= 学習で得られる典型結果を人手で再現したデモ用重み)
- Lesson T1〜T9 で「Token Embedding を見る」から「FFN を knowledge memory として解釈する」「複数 preset の attention 比較」まで段階的に学習。Lesson T1 冒頭で「重みは hand-crafted」と透明に明示する

詳細・段階定義は [`design.md`](design.md) §3「スコープ (Tier 定義)」を参照。

## 16 語固定 vocabulary

| ID | 単語 | 種別 |
|---|---|---|
| 0-2 | これ / それ / 私 | 指示詞・代名詞 |
| 3-6 | 猫 / 花 / 本 / 小説 | 名詞 |
| 7-9 | 大きい / 美しい / 人気 | 形容詞 |
| 10-11 | 読む / 好き | 動詞 |
| 12-14 | は / が / を | 助詞 |
| 15 | です | コピュラ |

これらで組める例文 (T=5):
- これ は 美しい 花 です — 「美しい」→「花」の attention
- 私 は 猫 が 好き — 「私」「猫」→「好き」の関係
- 大きい 猫 は 美しい です — 形容詞の連鎖
- 他 5 文 (`design.md` §6.2 参照)

## ディレクトリ (計画)

```
nn-anatomy-transformer/
├── README.md                      ← 本ファイル
├── docs/
│   ├── design.md                  ← 全機能仕様 (P0 完了)
│   ├── math-notes-transformer.md  (P8 で完備)
│   ├── lesson-plans-transformer.md(P8 で完備)
│   └── implementation-notes.md    (実装ログ)
├── src/
│   ├── index.html
│   ├── style.css
│   └── js/
│       ├── rng.js          (MLP 版から流用、PE 生成内部用)
│       ├── tokenizer.js    (P1) — 16 語固定 encode/decode
│       ├── matrix.js       (P1) — flat-array 行列演算
│       ├── presets.js      (P3) — デフォルト preset を inline
│       ├── model.js        (P3) — createTransformer / forward
│       ├── lessons.js      (P6) — Lesson T1〜T9
│       ├── view.js         (P4) — 行列ヒートマップ + pan/zoom
│       ├── explain.js      (P5) — 式の展開
│       └── controller.js
├── presets/                ← hand-crafted 重みの JSON
│   ├── japanese-mini-v1.json
│   ├── induction-head.json
│   ├── knowledge-lookup.json
│   └── random-init.json
├── tests/
│   ├── js/
│   ├── py/                 numpy 参照実装
│   └── fixtures/
├── tools/
│   ├── bundle.py           (MLP 版から流用)
│   ├── build_preset.py     設計書 §6.9 に従って hand-crafted preset JSON を構築 (numpy)
│   └── inline_preset.py    デフォルト preset を src/js/presets.js に展開
├── build/
├── Makefile
├── pyproject.toml          (numpy + pytest)
├── package.json
└── .gitignore
```

## 開発環境

- **uv 0.11+** (Python 環境)
- **Python 3.10+** (テスト・preset 構築用)
- **Node.js 20+** (`node --test`)
- **モダンブラウザ**

```bash
uv sync       # numpy + pytest が .venv に入る (数秒、PyTorch は使わない)
node --version
make help     # 利用可能なタスク
```

## ロードマップ (詳細は [`design.md`](design.md) §12)

| フェーズ | 内容 | 状態 |
|---|---|---|
| **P0** | 設計書 + プロジェクト雛形 | **完了** |
| P1 | matrix.js + tokenizer.js + numpy 参照実装 | **完了** (JS 27 / Python 13) |
| P2 | tools/build_preset.py + デフォルト preset 構築 (hand-crafted) | **完了** (Python 23 + preset 26.1 KB、self-check 16/16 pass) |
| P3 | Tier 1 model.js (forward only) + presets.js inline | **完了** (JS 39 / Python 36、計 75 件パス、attention fixture 数値一致) |
| P4 | Tier 1 view.js (静的描画) + pan/zoom + 最小 controller | **完了** (build = 53.6 KB、Tier 1 が画面に出る) |
| P5 | セルクリック → 式の展開 (explain.js) + Attention Map 常駐 | **完了** (build = 72.1 KB、X/Q/K/V/scores/attn/attnOut/Y の 8 種で 3 段板書) |
| P6 | Tier 1 Lesson T1-T5 + UI Lesson セレクタ | **完了** (build = 85.6 KB、JS 47/Python 36 = 83 件パス) |
| P7 | Tier 2 (Multi-Head + 残差 + LN + FFN) + Lesson T6-T8 | 未着手 |
| P5 | Tier 1 explain.js + interactivity | 未着手 |
| P6 | Tier 1 Lesson T1-T5 + 8 文セレクタ | 未着手 |
| P7 | Tier 2 (Multi-Head + 残差 + LN + FFN) + Lesson T6-T8 | 未着手 |
| P8 | Tier 3 (多 Block + 複数 Preset 切替) + Lesson T9 + ドキュメント完備 | 未着手 |

## 関連プロジェクト

- 姉妹: [nn-anatomy (MLP 版 v1)](https://github.com/naltoma/nn-anatomy) — 完成版。本プロジェクトの教材としての前提知識 (forward / backward / chain rule) を提供する。
