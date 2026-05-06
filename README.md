# NN Anatomy — Transformer

ブラウザで開くだけで動く、**Transformer の forward 動作を 1 セルずつ追える教材シミュレータ**。日本語の短い例文を入力に、Token Embedding → Multi-Head Attention → 残差接続 → LayerNorm → FFN → block 出力までの **すべての中間行列** を行列ヒートマップで観察できる。各セルをクリックすると `[段1] 一般形 → [段2] 当てはめ → [段3] 数値` の 3 段板書でその数値が「どう作られたか」をその場で展開する。

姉妹プロジェクト [nn-anatomy (MLP 版)](https://naltoma.github.io/nn-anatomy/) の哲学を Transformer に拡張したもので、大学情報系学部の **3〜4 年次以上** (深層学習・NLP の応用科目) の教材を想定している。

## 起動方法

セットアップ不要。`build/nn_sim_transformer.html` をブラウザで開くだけ。

```bash
# リポジトリをクローンしたら、
open build/nn_sim_transformer.html        # macOS
xdg-open build/nn_sim_transformer.html    # Linux
start build\nn_sim_transformer.html       # Windows
```

CDN や外部依存は無い (単一 HTML、~180 KB)。`file://` 経由でもオフラインで全機能が動く。

## v1 で見られるもの

**1 ブロック分の Transformer forward 全工程** が観察可能。本シミュレータの規模は次のとおり:

- **T = 5**: 入力 token の固定長 (= 1 文に必ず 5 トークンが並ぶ。語数の異なる例文には対応しない)
- **d_model = 16**: 各 token を表すベクトル次元
- **h = 2**: Multi-Head Attention のヘッド数 (各ヘッドの内部次元 d_k = d_model / h = 8)
- **d_ff = 32**: FFN 中間層のニューロン数

- **5 つの中央タブ**で内部状態を切り替え
  - Embed: token 埋め込み (W_E) と位置エンコーディング (W_P) と入力 X
  - Q/K/V: W_Q / W_K / W_V の重み行列と射影結果 (head 別表示)
  - Attn: scores と softmax 後の attention map (head 別)
  - Out: attnOut → Y → residual1 → ln1_out (残差 + LayerNorm)
  - FFN: W1 / ffn_h / W2 / ffn_out / residual2 / ln2_out (block 最終出力まで)
- **任意のセルをクリック**すると右の「式の展開」タブにその値の生成過程が 3 段で出る (例: `Q[T2 美しい, h0.d0] = 0.977 × +3.91 + … = +3.83`)
- **8 種類の例文セレクタ**で異なる文法関係を比較
- **Lesson セレクタ**で 10 本のステップ別カリキュラムをワンクリック起動

## Lesson 1〜10 (収録カリキュラム)

| Lesson | テーマ | 観察対象タブ |
|---|---|---|
| 1 | Token Embedding を見る (W_E[tokens] が主役) | Embed |
| 2 | 位置エンコーディング (W_P) を見る | Embed |
| 3 | Q (Query) projection を見る | Q/K/V |
| 4 | K (Key) projection を見る | Q/K/V |
| 5 | V (Value) projection を見る (identity の意義) | Q/K/V |
| 6 | Attention Score (Q·K^T / √d_k) | Attn |
| 7 | softmax と attention map | Attn |
| 8 | Multi-Head — 役割分担と同時並行 | Attn |
| 9 | 残差接続 + LayerNorm | Out |
| 10 | FFN (検出器ニューロン) と 1 ブロック完成 | FFN |

各 Lesson は **目的説明 (hint)** と **3 つの観察タスク (checks)** で構成。Lesson カードで「何を見るか / 何が見えるはずか / どう解釈するか」が文章で示され、ユーザーは中央タブで実際の値を確認しながら読み進められる。

## 16 語固定の日本語語彙

| ID | 単語 | 種別 |
|---|---|---|
| 0-2 | これ / それ / 私 | 指示詞・代名詞 |
| 3-6 | 猫 / 花 / 本 / 小説 | 名詞 |
| 7-9 | 大きい / 美しい / 人気 | 形容詞 |
| 10-11 | 読む / 好き | 述語 |
| 12-14 | は / が / を | 助詞 |
| 15 | です | コピュラ |

**収録例文 (各文ちょうど 5 トークン)**: 「これ は 美しい 花 です」「私 は 猫 が 好き」「私 は 本 を 読む」など 8 文。Lesson 内では「形容詞 → 名詞」「指示詞 → 名詞」「述語 → 名詞」のような bond パターンが attention で立つ様子を実際の数値で観察する。

## 設計上の特徴

- **forward 専用、学習機能は持たない**: 学習プロセスの理解は姉妹プロジェクト [nn-anatomy (MLP 版)](https://github.com/naltoma/nn-anatomy) で完結する想定。本シミュレータは「学習結果としての Transformer 内部で何が起きているか」だけに集中する
- **重みは hand-crafted preset**: SGD で学習させる代わりに、人手で意図的なパターン (例: 「形容詞 dim → bond 0」「名詞 ∧ 美しさ → 検出器 h12」) を埋め込んだ JSON を読み込む。これにより 8 サンプル × T=5 の小規模で学習させた場合に attention map がぼやけるリスクを回避し、教育目的の数値が常に再現される
- **JS forward と numpy 参照が 1e-12 で一致**: シミュレータの数値が「教科書通り」であることを CI で常時検証。Lesson 文中の数値も Python テスト (`tests/py/test_lesson_values.py`) で自動検証される

## ドキュメント

| ファイル | 内容 |
|---|---|
| [`docs/design.md`](docs/design.md) | 全機能仕様 (v0.4)、数式、preset 構築指針、多ブロック化の検討案 |
| [`docs/implementation-notes.md`](docs/implementation-notes.md) | 実装ログ (フェーズごとの完了内容と決定事項) |
| [`docs/initial-readme.md`](docs/initial-readme.md) | プロジェクト初期 (P0 設計フェーズ) の README をアーカイブ |

## 開発者向け

```bash
# Python (numpy + pytest) と Node 20+ を準備
uv sync
node --version

# テスト一式 (JS 50 + Python 62 件、~1 秒)
make test

# numpy 参照との fixture を再生成
make fixtures

# preset を再構築 (hand-crafted) → presets/japanese-mini-v1.json
make build-preset

# 単一 HTML を再ビルド → build/nn_sim_transformer.html
make inline-preset
make bundle

make help     # 利用可能なタスク一覧
```

詳細・実装方針は [`docs/design.md`](docs/design.md) を参照。

## 関連プロジェクト

- 姉妹: [nn-anatomy (MLP 版 v1)](https://github.com/naltoma/nn-anatomy) — 完成版。本プロジェクトの教材としての前提知識 (forward / backward / chain rule) を提供する。
