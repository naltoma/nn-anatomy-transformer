# 実装メモ (Implementation Notes) — Transformer 版

> このファイルは Transformer 版の開発記録 (フェーズごと、UI/UX 調整、決定事項など)。プロジェクト紹介・利用案内は [`../README.md`](../README.md)、設計の全文は [`design.md`](design.md) を参照。

姉妹プロジェクト [nn-anatomy](https://github.com/naltoma/nn-anatomy) (MLP 版 v1) の哲学を Transformer に拡張した教材。実装は新規だが、思想・テスト方針・bundle 戦略は v1 から継承。

---

## 進捗

| フェーズ | 内容 | 状態 |
|---|---|---|
| **P0** | 設計書 + プロジェクト雛形 | **完了** (v0.3 で hand-crafted preset 方針確定) |
| P1 | matrix.js + tokenizer.js + numpy 参照実装 | **完了** (JS 27 / Python 13 = 40 件、PyTorch 不採用) |
| P2 | tools/build_preset.py + デフォルト preset 構築 | **完了** (Python +23 件、preset 26.1 KB、self-check 16/16) |
| P3 | Tier 1 model.js (forward only) + presets.js inline | **完了** (JS +12 = 39 件、attention fixture と 1e-12 一致、計 75 件) |
| P4 | Tier 1 view.js (静的描画) + pan/zoom + 最小 controller | **完了** (bundle 53.6 KB、Tier 1 が画面に出る) |
| 中間 | レイアウトを上下分割に変更 | **完了** (controls / network / side の縦 3 段、bundle 54.1 KB) |
| P5 | Tier 1 explain.js + cell click/hover + Attention Map 常駐 | **完了** (bundle 72.1 KB、8 種行列の 3 段板書) |
| P6 | Tier 1 Lesson T1-T5 + UI Lesson セレクタ | **完了** (bundle 85.6 KB、JS +8 = 47 件、計 83 件) |
| P7 | Tier 2 (Multi-Head + 残差 + LN + FFN) + Lesson T6-T8 | 未着手 |
| P6 | Tier 1 Lesson T1-T5 + 8 文セレクタ | 未着手 |
| P7 | Tier 2 (Multi-Head + 残差 + LN + FFN) + Lesson T6-T8 | 未着手 |
| P8 | Tier 3 (多 Block + 複数 Preset 切替) + Lesson T9 + ドキュメント完備 | 未着手 |

## P0 (設計書 + プロジェクト雛形)

### 完了

- `docs/design.md` v0.2 作成 (= forward 専用 + 日本語 vocab + preset ベースの全面書き直し)
- プロジェクト雛形:
  - `README.md`: プロジェクト紹介、想定する完成像、ロードマップ
  - `Makefile`: serve / test / test-js / test-py / fixtures / train-preset / bundle / clean
  - `pyproject.toml`: PyTorch + pytest 依存
  - `package.json`: ES module 宣言のみ
  - `.gitignore`: v1 と同等
  - `src/js/rng.js`: v1 から流用 (PE 生成と内部の決定的乱数のみで使用、学習しない)
  - `tools/bundle.py`: v1 から流用、出力ファイル名と MODULE_ORDER を Transformer 用に書き換え
- ディレクトリ構成: `docs/` `src/js/` `tests/js/` `tests/py/` `tools/` `build/` `presets/`

### v1 からの方針継承

- 設計書のスタイル / 章立て (1. 目的 〜 16. リスク・制約 + 付録)
- mulberry32 + サブシード派生の乱数管理 (`rng.js` をそのままコピー)
- bundle.py の構造 (出力先と MODULE_ORDER だけ差し替え)
- Makefile のターゲット定義
- `.gitignore` の方針
- 「セルクリック → [段3] 数値展開」「ホバーで関連ハイライト」「3 段板書 ([段1] 一般形 → [段2] 当てはめ → [段3] 数値)」の哲学
- Lesson curriculum の hint + checks(3) + 補足 構造

### v1 からの方針変更

#### 大きな転換 (0.1 → 0.2 で決定)

- **forward 専用に絞る**: backward / 学習機能はシミュレータには含めない。MLP 版で chain rule を習得済みであることが前提の「次の教材」。
- **日本語 16 語固定 vocab + T=5**: a/b/c のような抽象トークンではなく、「これ / 花 / 美しい / です」など意味のある単語を採用。これにより attention map が言語的に解釈可能になる (例: 「美しい」→「花」が浮き上がる)。
- **Preset ベース運用**: 重みはランダム初期化ではなく、Python (PyTorch) で事前に学習させた preset JSON を読み込む。シミュレータ側は forward だけ。`tools/train_preset.py` が repo 開発側のスクリプトで preset を生成する。
- **Tier 4 段 → 3 段に縮約**: T1 = 1-head SA forward、T2 = 1-block 完成、T3 = 多 block + preset 切替。Backward + 学習だった Tier 3-4 を削除。
- **サイズ予算 350 KB → 250 KB**: backward / 学習コードがなくなったため。

#### 技術構成の変更

- **データ表現は flat Float64Array**: v1 はノードごとのネスト配列だったが、行列演算が中心になるので flat layout に変更
- **可視化パラダイム転換**: node-edge 図 → 行列ヒートマップ。各セルがクリック可能で、クリックで「式の展開」パネルにそのセルの 3 段板書を出すという哲学は継承
- **pan/zoom 必須**: 行列が画面に収まらないため、SVG viewBox の drag/wheel 操作で実装
- **Python 側は PyTorch**: v1 は NumPy だったが、Multi-Head Attention や LayerNorm の挙動を完全に揃えるため、テストと preset 学習に PyTorch を採用 (CPU 版で十分)

#### ボツ / 後送りにした案

- **Backward 可視化 (softmax の Jacobian)**: 当初 0.1 案では Tier 3 で扱う予定だったが、教育的負荷が大きすぎる + MLP 版で核は習得済み、として削除。v2 拡張余地として `design.md` §14 に残す。
- **学習タイムラプス**: Python 側で 100 step 刻みのスナップショットを取って preset 列として再生する案も検討したが、まずは「単一 preset の forward 観察」に集中し、Lesson T9 で 「複数 preset の比較」 (random vs trained) として代替。
- **Causal Mask**: GPT 系の必須要素だが Tier 1-3 では不要 (BERT 寄りに振る)。v2 で対応。

## P1 (matrix.js + tokenizer.js + numpy 参照実装) 完了

### 完了

- `src/js/tokenizer.js`: 16 語固定 vocab + encode/decode + 8 例文 (SAMPLE_SENTENCES)
- `src/js/matrix.js`: matmul / softmax / layerNorm / gelu / transpose ほか、flat-array ベース
- `tests/py/reference.py`: numpy で同等計算、float64 で fixture JSON 出力
- `tests/py/test_reference.py`: numpy 参照の sanity check (13 件)
- `tests/js/test_tokenizer.test.mjs`: vocab 整合性、encode/decode 往復、サンプル文 (12 件)
- `tests/js/test_matrix.test.mjs`: numpy fixture と 1e-12 オーダ一致 (15 件)
- 結果: **JS 27 件 / Python 13 件、計 40 件パス**

### v0.2 → v0.3 の方針転換

- `tests/py/reference.py` を当初 PyTorch 予定で作っていたが、`uv sync` が timeout (~200MB の torch CPU 版ダウンロード) したため numpy ベースに切り替え
- → そのまま「PyTorch 自体不要 (preset を学習しないので autograd が要らない)」と気付く
- v0.3 で hand-crafted preset 方針に転換、PyTorch 依存を完全削除
- LayerNorm の `numpy.var` (ddof=0 = 有偏) は PyTorch の `F.layer_norm` と一致するので、後から PyTorch を再投入することになっても 1e-12 オーダで揃う
- GELU は tanh 近似式 `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))` で JS / numpy 完全一致

## P2 (build_preset.py + デフォルト preset 構築) 完了

### 完了

- `tools/build_preset.py` (~280 行、numpy のみ): 設計書 §6.9 の表どおりに W_E / W_P / W_Q / W_K / W_V / W_O を書き込み、self-check (8 例文 × 2 関係 = 16 件) が softmax 後 0.5 以上で通ることを検証
- `presets/japanese-mini-v1.json` (26.1 KB): デフォルト preset として生成
- `tests/py/test_build_preset.py` (~150 行、23 テスト): embedding 各 dim、W_Q/W_K の bond、JSON 構造、parametrize で 8 例文すべての attention pattern を独立テスト

### 設計値

- CAT_VAL = 3.0 (1 次カテゴリ強度、PE のノイズ ±1 を上回るスケール)
- FEAT_VAL = 1.0 (2 次意味特徴)
- TRACE_VAL = 0.3 (識別痕跡)
- ATTN_W = 1.0 (W_Q/W_K の bond スケール)
- 3 つの bond: adj→noun (bond 0), pron→noun (bond 1), pred→noun (bond 2)

### Self-check 結果 (16 件すべて pass)

attention pattern の強さ範囲: 0.78〜0.96 (= 教育的に「明確に注目」と読める)
- pron→noun: 0.87〜0.96
- adj→noun: 0.78〜0.96
- pred→noun: 0.88

## P3 (model.js + presets.js inline) 完了

### 完了

- `tools/inline_preset.py`: presets/japanese-mini-v1.json を読み込んで src/js/presets.js (ES module、`export const DEFAULT_PRESET = {...}`) を生成。インデント無しで詰めて書き出すため 26.1 KB → 8.6 KB に圧縮
- `src/js/model.js` (~140 行): `createTransformer({ preset })` で preset から flat-array 構造体を構築、`forward(net, tokens)` で Embedding + PE → Q/K/V → scaled dot-product attention → output projection を計算。`reset(net)` で中間結果クリア
- `tests/py/generate_fixtures.py` を拡張: 8 例文を numpy で forward した結果 (X, Q, K, V, scores, attn, attnOut, Y) を `tests/fixtures/p3_attention.json` に出力
- `tests/js/test_model.test.mjs` (~200 行、12 テスト): 構造的テスト (createTransformer / forward / reset / 例外処理) + 8 例文すべての中間結果が numpy fixture と 1e-12 一致 + 主要 attention pattern (美しい→花、私→猫、読む→本) が 0.5 以上

### 設計上のポイント

- **net 構造は flat Float64Array**: `net.X[t * d_model + d]` のような index 計算でアクセス。ネスト配列より速く、後で WebGPU 化容易
- **blocks[] 構造**: Tier 2/3 で増えても同じ JSON 形式が使えるよう、Tier 1 でも `blocks[0].W_Q` のように管理
- **presetMeta**: kind / tier / designIntent を持ち、UI から「この preset は何か」を表示できる
- **数値一致は完璧 (1e-12 オーダ)**: numpy ↔ JS で同じ式を実装したので、softmax の数値安定化 (max 引き) も含めて完全一致

## P4 (view.js + pan/zoom + 最小 controller) 完了

### 完了

- `src/index.html` (~80 行): UI scaffold (controls / network / side パネル + タブ)
- `src/style.css` (~190 行): 3 領域 grid レイアウト、行列ヒートマップ用のセル / ラベル / pan/zoom コントロール
- `src/js/view.js` (~280 行): 行列ヒートマップ描画 (signed: 青/赤グラデーション、attn: 青濃淡)、Embed/Q/K/V/Attn/Out タブ切替、SVG viewBox の drag/wheel pan/zoom、Fit/100% ボタン
- `src/js/controller.js` (~130 行): 最小限のイベント配線 (Sample 切替、Forward、Reset、タブ切替、pan/zoom)
- `src/js/lessons.js` `src/js/explain.js`: P5/P6 用のスタブ (空 export だけ、bundle.py の MODULE_ORDER を満たすため)
- 結果: `make bundle` で `build/nn_sim_transformer.html` (53.6 KB) が生成、`file://` で開けば Tier 1 の Self-Attention 全 6 行列 (X, Q, K, V, scores, attn, attnOut, Y) がヒートマップとして見える

### 設計上の判断

- **タブ切替時に viewBox を全体表示にリセット**: タブごとに行列のサイズが違う (Embed は T×16、Attn は T×T) ので、切り替え時に「全部見える」状態に毎回戻す。pan/zoom はそのタブ内での探索用。
- **forward 前でも Embed タブに W_E + W_P を見せる**: previewSampleInput() で X だけ事前計算する。学生が「forward 前は何も無し、forward 後に全部出る」よりも「最初から入力埋め込みは見える」のほうが流れを追いやすい。
- **signed セル色**: 青=負 / 白=0 / 赤=正、明度は |v| / |v|_max で正規化。行列ごとに max を取るので、値の絶対値が小さい行列でも違いが見える。
- **attn セル色は単一色青濃淡**: 0..1 の確率分布なので、青の単色の濃さで表現。値が 0.5 以上のセルはテキストを白にして可読性を確保。
- **pan/zoom は SVG viewBox 操作で実装**: 50 行程度で完結。マウスドラッグで pan、ホイールで zoom (マウス位置中心)。

### バンドル内訳 (53.6 KB)

| モジュール | 寄与 |
|---|---|
| presets.js (inline preset JSON) | ~9 KB |
| view.js | ~10 KB |
| style.css | ~5 KB |
| model.js | ~4 KB |
| matrix.js | ~3 KB |
| controller.js | ~4 KB |
| その他 (rng, tokenizer, lessons stub, explain stub, index.html) | ~18 KB |

予算 250 KB に対して大幅に余裕がある状態。P5 (explain.js) で式テンプレートが入っても 80-100 KB 程度で収まる見込み。

## レイアウト変更 (横並び → 縦 3 段)

行列は横長 (5×16 = 800 px) なので、右側 480 px の補助パネルが幅を圧迫していた問題を解消。`style.css` の grid を `1fr × (auto / 1fr / 320px)` に変更し、`controls / network / side` の縦並びにした。これでメインビューが画面幅をフルに使える。bundle 53.6 → 54.1 KB (+0.5 KB のみ、コメント増分)。

## P5 (explain.js + cell click/hover interactivity) 完了

### 完了

- `src/js/explain.js` (~280 行): 各行列のセルクリック時に表示する 3 段板書 HTML テンプレート
  - **8 種類の行列** に対応: X (Embedding), Q, K, V, scores, attn, attnOut, Y
  - 各行列で `[段1] 一般形 → [段2] 当てはめ → [段3] 数値展開` の 3 段構成
  - 数値展開は「W_Q[d,k] = 0 の項はスキップ」など、hand-crafted の bond 設計に合った見やすい表示
  - softmax の説明では行ごとの max 引きと exp を 5 サンプル分すべて展開
- `view.js` 側の修正:
  - `createView(svg, net, { onCellClick })` を追加
  - 各セル `<rect>` に click ハンドラを登録、cursor: pointer 設定
  - 選択中のセルは `cell-selected` クラスで強調 (orange ストローク)
  - hover で薄い orange ストロークが付くインタラクション
- `controller.js` 側の修正:
  - `onCellClick({ matrix, row, col })` で `state.selection` を更新 → renderExplain → 自動で「式の展開」タブにフォーカス
  - Forward / Reset / Sample 切替時に refresh
- `renderAttentionMapHtml()`: 補助パネルの「Attention Map」タブに T×T のテーブルとして常駐表示。中央パネルがどのタブを見ていても attention map が常に確認できる
- `style.css`: 数式表示用 (.math, .mnum, .mop)、選択セル強調 (.cell-selected)、Attention Map テーブル (.attmap-table)

### 設計上の判断

- **「式の展開」タブが自動でアクティブに**: セルクリック直後に式が見えないと意図が伝わらないので、クリック → タブ自動切替する流れに
- **0 の項を省略**: hand-crafted preset では W_Q が「ほとんど 0、bond dim だけ 1」なので、全項を展開すると見づらい。0 を skip して「効いている項だけ書く」と式の意味が明確になる
- **softmax は max 引きを明示**: `e^{score_i,j − max}` という形で書くことで、数値安定化テクニックが教育的に伝わる
- **Attention Map を補助パネルに常駐**: 中央パネルで Embed / Q-K-V / Out など別の行列を見ていても、attention の概観は常に右下で確認できる

### バンドル増分内訳 (54.1 → 72.1 KB = +18 KB)

| 寄与 | 増分 |
|---|---|
| explain.js (3 段板書テンプレ) | ~10 KB |
| style.css (式表示・attmap・selected) | ~3 KB |
| controller.js (クリック配線、refresh) | ~1 KB |
| view.js (handlers, setSelection, click) | ~1 KB |
| その他 | ~3 KB |

予算 250 KB 内で大幅余裕。Tier 2 で multi-head + LN + FFN を追加しても 130 KB 程度で収まる見込み。

## P6 (Lesson T1-T5 + Lesson セレクタ) 完了

### 完了

- `src/js/lessons.js` (~110 行): Lesson T1-T5 を `LESSONS` 配列 + `getLesson(id)` で公開
  - T1: Token Embedding を見る (sample=0, viewTab=embed, runForward=false)
  - T2: Positional Encoding を見る (sample=0, viewTab=embed, runForward=false)
  - T3: Q/K/V projection を見る (sample=0, viewTab=qkv, runForward=true)
  - T4: Attention Score (sample=0, viewTab=attn, runForward=true)
  - T5: softmax と attention map (sample=2「私 は 猫 が 好き」, viewTab=attn, runForward=true)
- `src/index.html`: Lesson セレクタを Sample の左に追加 (`<select id="lesson">`)
- `src/js/controller.js`:
  - `LESSONS` から `<option>` を生成
  - `onLessonChange()`: Sample 切替 + viewTab 切替 + 必要なら自動 Forward + 補助パネル「レッスン」タブを active
  - `renderLessonCard()`: hint + checks(3) を補助パネルに描画
  - `activateSideTab()` / `activateViewTab()` のヘルパで重複を整理
  - explain.js から `escapeHtml` を import (top-level 名前衝突を回避)
- `src/style.css`: `.lesson-card` 用のヒント枠 + checks リスト + `white-space: pre-line` (将来 `\n\n💡 補足` 区切りに対応)
- `tests/js/test_lessons.test.mjs` (~70 行、8 テスト):
  - LESSONS が T1〜T5 の 5 件、各々が必須フィールドを持つ
  - sampleIdx は SAMPLE_SENTENCES の範囲内、viewTab は許容セット
  - getLesson の既知/未知 id 動作
  - T3 が viewTab=qkv & runForward=true、T4/T5 が attn、T5 のサンプルが「私 は 猫 が 好き」

### 各 Lesson の中身

- **T1 (Embed)**: 各 dim に意味が割り当ててあること (dim 0=is_noun, 1=is_adjective, 8=美しさ特徴 etc.) を実機で確認。「美しい」と「花」が dim 8 で意味的に近い、など。
- **T2 (PE)**: PE は固定 (sin/cos)、token を変えても W_P 部分は同じ、というのを Sample 切替で観察。共通トークン (私/は) が同じ位置にあれば X が完全一致することも確認。
- **T3 (Q/K/V)**: hand-crafted bond により Q[T2「美しい」] が dim 0 だけ立つ、K[T3「花」] が dim 0/1/2 立つ、V = X (identity)、をセル数値で確認。
- **T4 (scores)**: scores[T2, T3] だけ +0.25 になり、他の scores 行 (助詞/コピュラの Q) はほぼ全列 0、という対比。
- **T5 (softmax)**: 「私 は 猫 が 好き」で「私 → 猫: 0.96」「好き → 猫: 0.88」を観察。助詞の attn 行はフラット (= 各 ~0.2)。`💡 補足` で softmax の集中性 (scores 0.25 vs 4 で attn が大きく変わる) を解説。

### 設計上の判断

- **Lesson 切替で View タブも自動切替**: T3 を選んだら自動で Q/K/V タブに飛ぶ。学生が「どこを見ればいいか」迷わないように。
- **Sample 手動変更で Lesson を外す**: ユーザが Sample を独自に切り替えたら Lesson の文脈は外れて「— (custom)」に戻る。Lesson の sampleIdx と Sample の現在値が食い違うと混乱するので。
- **runForward フラグ**: T1, T2 (Embed 観察) では Forward しない (X だけで足りる)。T3 以降は forward を必須に。
- **`escapeHtml` を explain.js から export**: MLP 版 v1 でも同じ問題があった。bundle.py の top-level 名前衝突検出が効いて build 時に検出できたので素直に共有化。

### バンドル増分内訳 (72.1 → 85.6 KB = +13.5 KB)

| 寄与 | 増分 |
|---|---|
| lessons.js (Lesson T1-T5 hint + checks 計約 9 KB の日本語) | ~9 KB |
| controller.js (Lesson 配線、ヘルパ整理) | ~2 KB |
| index.html (Lesson セレクタ追加) | ~0.3 KB |
| style.css (.lesson-card 関連) | ~1 KB |
| その他 | ~1 KB |

予算 250 KB 内で大幅余裕。Tier 2 (multi-head + LN + FFN + Lesson T6-T8) で +50-70 KB を見込んでも 150-160 KB 程度。

### 次のステップ (P7: Tier 2)

`src/js/model.js` に multi-head + 残差 + LayerNorm + FFN を追加し、設計書 §6.4-§6.6 の数式に従って forward を拡張。`build_preset.py` で Tier 2 用の重み (W_Q/W_K/W_V を h=2 に分割、LN の γ/β、FFN の W1/b1/W2/b2) を hand-crafted で構築。`view.js` に FFN タブと head 切り替え UI を追加。Lesson T6-T8 を作成 (Multi-Head の比較、残差+LN、FFN の knowledge memory 観察)。
