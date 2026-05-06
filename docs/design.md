# NN Anatomy — Transformer 版 設計書

版: 0.4 (2026-05-03、1 ブロック完成版を反映)
姉妹プロジェクト: [nn-anatomy](https://github.com/naltoma/nn-anatomy) (MLP 版 v1)。

変更履歴:
- 0.1 (2026-04-29) 初稿。Tier 1-4、Self-Attention 数式、行列ヒートマップ、Backward/学習込み。
- 0.2 (2026-04-29) 方針転換: forward 専用、日本語 16 語彙、事前学習済み preset (PyTorch で学習)。
- 0.3 (2026-04-30) 再方針転換 (preset の作り方):
  - 事前 学習 済み preset → 事前 構築 (hand-crafted) preset に変更。
  - 理由: 8 サンプル × T=5 では SGD が attention を綺麗に分化させる保証がなく、PE 暗記でショートカットされるリスクが高い。教材としては人手で意図的なパターン (例: 「美しい→花」が attention 0.7+) を作り込むほうが、教育目的が明確で再現性も完璧。
  - PyTorch 依存を完全削除 (numpy のみで参照実装と preset 構築の両方をまかなう)。
  - `tools/train_preset.py` → `tools/build_preset.py` に改名。autograd 学習ではなく、設計書の表に従って重みを直接書き込む素朴なスクリプトに。
  - Lesson T1 の冒頭で「重みは学習結果ではなく hand-crafted」と透明に明記。
- **0.4 (2026-05-03) 1 ブロック完成版を反映**:
  - 実装が P0〜P7c (Tier 2: Multi-Head + 残差 + LayerNorm + FFN) まで完了。
  - Lesson 数を当初想定の 9 から **10 本** に拡大: Q/K/V 観察を Lesson 3/4/5 に分割した経緯を反映。Lesson 9 は残差 + LN、Lesson 10 は FFN。
  - 「Tier 2 = h=2, d_k=8」「3 bond を 2 head に分散 (head 0 = adj/pron→noun, head 1 = pred→noun)」を確定設計として記述。
  - Lesson 文中の数値 (scores 値、attn 値、ffn_h 検出器発火など) を Python テストで自動検証する仕組みを追加 (`tests/py/test_lesson_values.py`)。
  - 多ブロック化 (P8) を「現状を踏まえた検討案」として §3.4 に追記。

---

## 1. 目的

Transformer の **inference (推論) 時の動作** を、**「行列のセル 1 個ずつ何が起きているか」** という粒度で初学者に理解させる。具体的には次の 3 点を、日本語の短い例文 (例: 「これ は 美しい 花 です」) を入力に手を動かしながら確認する。

1. **Self-Attention の核**: Q, K, V がどのように作られ、Q·K^T が attention score になり、softmax で確率化された後 V との重み付き和 (attention output) になるか
2. **Multi-Head の意味**: 複数の head が並列に動き、独自の attention map を作って後で連結されるか (head ごとに違う言語的関係を担当する)
3. **Block 単位の流れ**: Attention → 残差 + LayerNorm → FFN → 残差 + LayerNorm

学習 (Backward / Update / Loss) は本シミュレータでは扱わない。**MLP 版 v1 ([nn-anatomy](https://github.com/naltoma/nn-anatomy)) でその核は習得済みである** ことを前提にする (姉妹プロジェクトの位置付け)。

重みはシミュレータでランダム初期化するのではなく、**Python (numpy) 側で事前に hand-crafted (人手構築) で書き込んだ preset JSON を読み込む** ことで attention map が言語的に意味を持つ状態を保証する。学生は学習プロセスではなく **典型的な学習結果として「こうなっていてほしい」内部の動作観察** に集中する。Transformer の学習プロセス自体は MLP 版 (`nn-anatomy`) で習得した chain rule を Self-Attention に拡張すれば導出できるため、本シミュレータでは扱わない。

## 2. 想定ユーザーと前提知識

| 項目 | 想定 |
| --- | --- |
| 対象 | 大学情報系学部の **3〜4 年次 / 修士 1 年** (深層学習・自然言語処理の応用科目) |
| 必須前提 | 線形代数 (行列積、内積、転置)、確率の基礎 (softmax)、MLP 版 (`nn-anatomy`) で習得した forward / backward / chain rule |
| 推奨前提 | NumPy / PyTorch の基本操作 |
| 不要 | フレームワーク利用経験、CUDA、tokenization 経験、Transformer の事前知識 |
| 言語 | v1 は日本語のみ (en は将来) |

MLP 版を使った後の **次の教材** として位置付ける。Lesson 文中で随所に MLP 版へ参照を貼り、「Lesson5 で見た連鎖律と同じ理屈で、Transformer も学習できる (詳細は `tools/train_preset.py` 参照)」のような連続性を持たせる。

## 3. スコープ (Tier 定義)

教材としての段階を 3 ティアに分け、Tier 1 だけでも独立教材として成立するように設計する。**いずれも forward (推論) のみ** で、学習は repo 開発側のスクリプトで preset 生成済みの状態を前提とする。

### Tier 1: Single-Head Self-Attention の Forward (履歴: 直接 Tier 2 を実装したため独立リリースは無し)

当初の MVP 想定。日本語例文 1 つを通して Q/K/V → Attention → Output を観察する。

| 項目 | 仕様 |
| --- | --- |
| シーケンス長 | T = 5 (日本語短文に合わせる: 「これ は 美しい 花 です」) |
| 埋め込み次元 | d_model = 16 |
| ヘッド数 | h = 1 |
| ヘッド次元 | d_k = d_model / h = 16 |
| 語彙 | 16 種 (§6.1 で定義) |
| Block 数 | N = 1 |
| 計算範囲 | Token Embedding + Positional Encoding → Q/K/V projection → Attention → Output projection |
| 重み | 事前構築 (hand-crafted) preset を JSON 読み込み |

注: 実装は P7a で直接 Tier 2 (h=2) に移行したため、Tier 1 (h=1) としての独立リリースは無い。Lesson 1〜5 は Tier 2 preset の上で「単 head の動作」を観察する形で実現している。

### Tier 2: 1 Block 完成 (Multi-Head + Residual + LayerNorm + FFN) ← v0.4 時点の実装範囲

Transformer Block 1 段の完全像。実用 Transformer の構造を全部見せる。**v0.4 時点で完成**。

| 追加要素 | 仕様 | 担当 Lesson |
| --- | --- | --- |
| ヘッド数 | h = 2 (d_k = 8 ずつ)。bond 0/1 を head 0、bond 2 を head 1 に分散 | Lesson 8 |
| 残差接続 (1 回目) | Self-Attention 出力 + 入力埋め込み | Lesson 9 |
| LayerNorm 1 | 残差後の各位置で正規化 (γ=1, β=0 固定) | Lesson 9 |
| FFN (Feed-Forward) | 2 層 MLP (d_model → d_ff=32 → d_model)、活性化は GELU。検出器ニューロン設計 | Lesson 10 |
| 残差 + LayerNorm の 2 回目 | FFN 出力 + その入力 | Lesson 10 |

UI: 「FFN タブ」を追加し、W1 / ffn_h / W2 / ffn_out / residual2 / ln2_out を縦に並べて観察可能。FFN 中間層を「knowledge memory」として hand-crafted の検出器が発火する様子を Lesson 10 で見せる。

### Tier 3: 多 Block + 複数 Preset 切替 (検討案)

実用に近い状態。**v0.4 時点で未着手**。各要素は独立に進められる。

| 追加要素 | 仕様 |
| --- | --- |
| Block 数 | N = 2 (最大 4) |
| Preset 切替 | UI で複数の preset (japanese-mini-v1, induction-head, knowledge-lookup, random-init 等) を切り替えて attention map を比較 |
| Layer 切り替え UI | タブで切り替え |
| Export / Import | 重みを JSON で保存・復元 |

### 3.4 多ブロック化 (Lesson 11) で得られる学習体験 — 検討案

「Tier 3 の中で最初に着手するなら何が一番効くか」の検討メモ。

**Transformer の本質である「層を積めば積むほど抽象的な表現が得られる」性質** は、1 ブロックでは見せ切れない。Lesson 1〜10 で観察したのは「1 段の forward でどれだけのことが起きるか」であり、層を 2 段以上に増やすと **次のような新しい観察軸** が出てくる。

#### 3.4.1 期待できる教育的観察

1. **抽象度の積み重ね**
   Block 1 で「形容詞 → 名詞」のような **直接的な統語関係** を集約。Block 2 はその「混ざった状態」を入力にして、さらに広い文脈関係を形にする。例えば「私 は 猫 が 好き」では Block 1 で「私 → 猫」「好き → 猫」が並列に立つ。Block 2 ではその両方を持つ「猫」自身が「私と好きの両方を浴びた状態」になっており、attention で「私 ↔ 好き」のような **間接的な関係** に発火させられる可能性がある (主語と述語が直接 attention を張る経路を作れる)。

2. **検出器の階層性 (FFN の knowledge memory が深くなる)**
   Block 1 の FFN h12〜h15 は「名詞 ∧ 美しさ」のような **1 段階の AND** を検出。Block 2 の FFN は「Block 1 の出力 (= attention で集めた + AND で強化された) 」を入力にするので、**「美しい花が attention で集めた状態 ∧ それを参照する指示詞」** のような 2 段階の組み合わせ検出を hand-crafted で組み込める。Lesson 11 の core になる観察。

3. **残差接続の威力 (深さに対する安定性)**
   Block 1 → Block 2 で残差を 4 回経由する (each block 内で 2 回 × 2 block)。それでも T0「私」 の元埋め込み情報 (`is_pronoun = +3`) が ln4_out までほぼ完全に保たれることを観察。**「層を積んでも token のアイデンティティは消えない」** という性質が数値で見える。MLP 版 Lesson4 の勾配消失の話と対照的に、forward 方向では残差が情報を運ぶ役割を担っている。

4. **同じ token のベクトルが「層を経るごとに豊かになる」進化**
   T0「私」を Block 0 入力 (= X)、Block 1 出力、Block 2 出力で並べると、d2 (指示詞 dim) は ≈ +3 のまま保たれるが、d0 (名詞 dim) は Block 1 で「猫」を吸収して +1.91、Block 2 でさらに「猫の特徴」も吸収して +X.X 付近まで増える、というベクトル変化を直接見られる。**「token の意味は周りから集めた情報で次第に豊かになる」** という Transformer の感覚を最も短い経路で伝えられる。

#### 3.4.2 hand-crafted preset の設計案

Block 2 を「Block 1 の出力をさらに処理する」に作るには、いくつかの選択肢がある:

##### 案 A: Block 2 = identity-like (まずは安定動作だけ見せる)

- W_Q^(2), W_K^(2), W_V^(2) = ほぼ identity (微小な摂動)
- FFN^(2) も恒等的
- Block 2 通すと **ほとんど値が変わらない** ことを観察 → 「層を積んでも壊れない」性質と「無駄な層を積むだけでは情報は増えない」両方が見える

教育的価値: 残差接続の **情報保持** の側面に集中。実装も最小。

##### 案 B: Block 2 = 反対方向の bond (主語 ↔ 述語 を逆向きに張る)

- Block 1 で「修飾語 → 名詞」「指示詞 → 名詞」「述語 → 名詞」を集約
- Block 2 で「名詞 → 述語」(逆向き) の bond を 1 本追加し、名詞が「自分が参照されている述語」を逆引き

教育的価値: bond の Q/K 役割の **学習対称性** (attention を双方向に張る効果) を見せられる。preset 設計は単純。

##### 案 C: Block 2 = 高次特徴の AND 検出 (Lesson 11 の主軸候補・推奨)

- Block 1 出力 ln2_out (= 検出器が発火済み) を入力にする
- Block 2 の FFN で **Block 1 で検出済みの特徴の AND** を検出
- 例: h_k = "「美しい名詞」を attention で受け取った指示詞" 検出器
  - Block 1 で T0「これ」が attention で T3「花」(名詞∧美しさ) を取り込んでいる場合のみ発火
  - W1[d8 (= 美しさ強化済み), k] = +0.7, W1[d2 (= 指示詞), k] = +0.7, b1[k] = -0.5

教育的価値: **「FFN の検出器も層を積めば階層化する」** という、Transformer の表現学習の本質を hand-crafted で見せる。preset 設計は §6.9.4 の延長で組める。

#### 3.4.3 Lesson 11 の構成案 (sampleIdx は #3 を引き継ぎ)

- **Task 1: 同じ token が層を経て変化する様子**
  T0「私」の X → ln2_out (Block 1 出力) → ln4_out (Block 2 出力) を 3 段で並べる。d0 (名詞 dim) が層を経るごとに「猫」由来の情報を集めてどう変わるかを観察。d2 (指示詞 dim) は残差で保たれることを対比。

- **Task 2: Block 2 の attention が Block 1 と違う関係を見る**
  案 B または案 C 採用時、Block 2 の attention タブで **Block 1 とは違う bond pattern** が立つことを確認。例えば「Block 1 では『私』→『猫』だったのが、Block 2 では『猫』→『好き』が立つ」のような変化。

- **Task 3: FFN の検出器が階層化している様子 (案 C 採用時)**
  Block 1 FFN の h12「名詞∧美しさ」と Block 2 FFN の h_k「美しい名詞を attention で集めた指示詞」を見比べ、h_k が Block 1 で発火している token に限って発火することを確認。

#### 3.4.4 実装コスト見積り

| 作業 | 規模 |
| --- | --- |
| build_preset.py: 第 2 ブロック分の重み構築 | 中 (案 A は小、案 C は ~50 行) |
| model.js: 多ブロック対応 (Block ループ) | 小 (現状のコードを `for (block of blocks)` で囲うだけ) |
| view.js: Block 切替 UI | 中 (タブまたはセレクタ追加、行列ヒートマップは流用) |
| explain.js: 「Block N の」修飾を全 render 関数に通す | 中 |
| Lesson 11 の checks (3 件) + 数値検証テスト | 小 |
| numpy 参照 + fixtures 更新 | 小 |

**最初の着手としては案 A から始めて多ブロック化のフレームを作り、案 C で教育コンテンツを乗せる**、という 2 段階が現実的。

## 4. 画面レイアウト (ワイヤー)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ [A] コントロールバー                                                    │
│ Lesson [▼] / Sentence [▼: これは美しい花です ▼] / Preset [▼] /         │
│ heads [1] / blocks [1] / Speed [1.0×]                                  │
│ [Forward]  [Reset]                              phase: [forward done]   │
│ [Import Preset]  [Export]                                              │
├─────────────────────────────────────────────────────────────────────────┤
│ [B] ネットワーク・ビュー (中央、最大領域)              [C] 補助パネル   │
│                                                                         │
│  ┌──────────────────────────────────────────┐  ┌──────────────────┐    │
│  │  Tab: [Embed][Q/K/V][Attn][Out][FFN][Final]│  │ Tab:             │    │
│  │  ┌──────────────────────────────────────┐ │  │ [式の展開]       │    │
│  │  │ T0 これ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  (16 cells)│ │  │ [Attention Map]  │    │
│  │  │ T1 は    ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒          │ │  │ [レッスン]       │    │
│  │  │ T2 美しい▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒          │ │  │ [Preset 情報]    │    │
│  │  │ T3 花    ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒          │ │  │ [イベントログ]   │    │
│  │  │ T4 です  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒          │ │  ├──────────────────┤    │
│  │  └──────────────────────────────────────┘ │  │                  │    │
│  │   Q (T×d_k):              K (T×d_k):     │  │  選択中: Q[2,3]  │    │
│  │   ┌─────────────┐         ┌─────────────┐│  │                  │    │
│  │   │             │         │             ││  │  [段1] 一般形    │    │
│  │   └─────────────┘         └─────────────┘│  │  Q = X · W_Q     │    │
│  │   Attention Score Q·K^T:                  │  │                  │    │
│  │   ┌─────────────┐                         │  │  [段2] 当てはめ  │    │
│  │   │ これ→これ:0.1│ ← 行/列に単語ラベル   │  │  Q[T2,3] = ...   │    │
│  │   │ これ→花 :0.5│                         │  │                  │    │
│  │   └─────────────┘                         │  │  [段3] 数値      │    │
│  │  [pan/zoom コントロール] [全体表示]      │  │  Q[T2,3] = +0.412│    │
│  └──────────────────────────────────────────┘  └──────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

3 領域構成は MLP 版と同じ。中央 [B] のタブで「ネットワーク内のどの段階を見るか」を切り替える。pan/zoom は中央の各タブの内容に対して効く。

### 中央パネルのタブ詳細

| タブ | 表示内容 | サイズ感 (Tier 1, T=5, d_model=16) |
|---|---|---|
| Embed | 入力 token (単語) と embedding + positional 行列 | 5 トークン × 16 セル |
| Q/K/V | Q, K, V の 3 行列、それぞれ T×d_k | 5×16 が 3 個 |
| Attn | Attention Score (Q·K^T / √d_k) と softmax 後の attention map | 5×5 が 2 個 |
| Out | Attention Output (softmax · V) と output projection 後 | 5×16 が 2 個 |
| FFN | 2 層 MLP の中間と出力 (Tier 2 以降) | 5×32, 5×16 |
| Final | ブロック出力 (残差 + LayerNorm 後) | 5×16 (Tier 2 以降) |

各 token 行のヘッダには **token ID と単語ラベル** (例: `T0 これ`) を表示。Attention map の行・列ヘッダにも単語ラベルが入るので、「これ → 花」のような関係が文字どおり読める。

## 5. データモデル

### 5.1 ネット構造

```js
// Tier 1: Single-Head SA, forward only
const net = {
  // ハイパパラメータ
  T: 5,              // sequence length
  d_model: 16,
  h: 1,              // heads
  d_k: 16,
  vocabSize: 16,
  N: 1,              // blocks (Tier 3 で増える)

  // Vocabulary (固定、§6.1 参照)
  vocab: ["これ", "それ", "私", "猫", "花", "本", "小説",
          "大きい", "美しい", "人気", "読む", "好き",
          "は", "が", "を", "です"],
  word2id: {...},    // 逆引き (Object<string,int>)

  // Preset から読み込んだパラメータ (固定)
  W_E: Float64Array(vocabSize * d_model),    // token embedding
  W_P: Float64Array(T * d_model),            // positional encoding (固定)
  W_Q: Float64Array(d_model * d_k),
  W_K: Float64Array(d_model * d_k),
  W_V: Float64Array(d_model * d_k),
  W_O: Float64Array(d_k * d_model),          // output projection
  presetName: "japanese-mini-v1",            // ロードした preset の識別子
  presetMeta: {...},                          // 学習元の例文集など、表示用メタ情報

  // Forward 中間結果 (シミュレータが計算する側)
  tokens: Int32Array(T),                      // 入力トークン ID 列
  X: Float64Array(T * d_model),               // embedding + positional
  Q: Float64Array(T * d_k),
  K: Float64Array(T * d_k),
  V: Float64Array(T * d_k),
  scores: Float64Array(T * T),                // (Q·K^T) / sqrt(d_k)
  attn: Float64Array(T * T),                  // softmax(scores)
  attnOut: Float64Array(T * d_k),             // attn · V
  Y: Float64Array(T * d_model),               // attnOut · W_O

  // メタ
  phase: "idle" | "forward",
};
```

すべて Float64Array の 1 次元配列で持ち、`get(net.Q, t, k) = net.Q[t * d_k + k]` のような index 計算で 2D アクセスする。MLP 版がネスト配列だったのと違い、行列積を効率良く回すために flat layout を採用する。

### 5.2 Tier 2 以降の拡張

```js
// Tier 2: Multi-Head + Residual + LN + FFN
//   W_Q, W_K, W_V を head ごとに保持: shape = (h, d_model, d_k)
//   さらに LayerNorm パラメータ (γ, β) と FFN の W1, b1, W2, b2

// Tier 3: 多 Block
//   net.blocks[l] = { W_Q, W_K, ..., FFN, LN1, LN2 } のように層ごとに

const net = {
  T, d_model, h, d_k, d_ff, N, vocabSize, vocab, word2id,
  W_E, W_P,
  blocks: [
    {
      W_Q, W_K, W_V,    // 形状 (h, d_model, d_k)
      W_O,
      LN1: { gamma, beta },
      LN2: { gamma, beta },
      FFN: { W1, b1, W2, b2 },
    },
    ...
  ],
  presetName, presetMeta,
  // Forward 中間も block ごとに保持
};
```

## 6. 計算仕様

### 6.1 Vocabulary (16 語固定)

シミュレータは以下 16 語の語彙を持ち、入力は必ずこの語彙から選ばれた T=5 トークン列とする。

| ID | 単語 | 種別 |
|---|---|---|
| 0 | これ | 指示詞 |
| 1 | それ | 指示詞 |
| 2 | 私 | 代名詞 |
| 3 | 猫 | 名詞 (生物) |
| 4 | 花 | 名詞 (植物) |
| 5 | 本 | 名詞 (物体) |
| 6 | 小説 | 名詞 (テキスト) |
| 7 | 大きい | 形容詞 |
| 8 | 美しい | 形容詞 |
| 9 | 人気 | 形容詞的名詞 |
| 10 | 読む | 動詞 |
| 11 | 好き | 形容詞動詞 |
| 12 | は | 助詞 (主題) |
| 13 | が | 助詞 (主格) |
| 14 | を | 助詞 (対格) |
| 15 | です | コピュラ |

### 6.2 サンプル文 (T=5)

UI の Sentence セレクタで切り替え可能。デフォルトの preset (`japanese-mini-v1`) はこれら 8 文の next-token prediction で訓練される (シミュレータでは結果のみ使用)。

| # | 文 | tokens | 観察できる attention 関係 |
|---|---|---|---|
| 1 | これ は 美しい 花 です | [0, 12, 8, 4, 15] | 「美しい」→「花」、「これ」→「花」 |
| 2 | それ は 人気 小説 です | [1, 12, 9, 6, 15] | 「人気」→「小説」、「それ」→「小説」 |
| 3 | 私 は 猫 が 好き | [2, 12, 3, 13, 11] | 「私」→「好き」、「猫」→「好き」 |
| 4 | 私 は 本 を 読む | [2, 12, 5, 14, 10] | 「私」→「読む」、「本」→「読む」 |
| 5 | 私 は 小説 を 読む | [2, 12, 6, 14, 10] | (4) と同パターン、目的語のみ違う比較教材 |
| 6 | 大きい 猫 は 美しい です | [7, 3, 12, 8, 15] | 「大きい」→「猫」、「美しい」→「猫」 |
| 7 | 美しい 花 は 人気 です | [8, 4, 12, 9, 15] | 「美しい」→「花」、「人気」→「花」 (連鎖) |
| 8 | これ は 大きい 本 です | [0, 12, 7, 5, 15] | 「大きい」→「本」、「これ」→「本」 |

### 6.3 Tokenizer

固定語彙のため非常に単純:

```
encode(["これ", "は", "美しい", "花", "です"]) → [0, 12, 8, 4, 15]
decode([0, 12, 8, 4, 15]) → ["これ", "は", "美しい", "花", "です"]
```

長さ T=5 と一致しないとエラー。サンプル文は事前に encode 済み Int32Array としても保持。

### 6.4 Token Embedding + Positional Encoding

```
X[t, :] = W_E[tokens[t], :] + W_P[t, :]      (t = 0..T-1)
```

W_P は **固定値** として sin/cos の位置符号 (元論文の方式) を初期化時に書き込む。preset には含めない (実行時に毎回生成、再現性に問題なし)。

### 6.5 Single-Head Self-Attention (Tier 1)

```
Q = X · W_Q                  shape: (T, d_k)
K = X · W_K                  shape: (T, d_k)
V = X · W_V                  shape: (T, d_k)

scores[i, j] = (Q[i, :] · K[j, :]) / sqrt(d_k)        shape: (T, T)
attn[i, :]   = softmax(scores[i, :])                  shape: (T, T)
attnOut[i, :] = sum_j(attn[i, j] · V[j, :])           shape: (T, d_k)

Y[i, :] = attnOut[i, :] · W_O                         shape: (T, d_model)
```

- スケーリング `1/sqrt(d_k)` は Tier 1 から含める。
- causal mask は Tier 1 では **なし** (双方向 attention)。BERT 寄りの解釈。

### 6.6 Multi-Head Self-Attention (Tier 2)

各 head h について:

```
Q_h = X · W_Q[h]    K_h = X · W_K[h]    V_h = X · W_V[h]
attnOut_h = Attention(Q_h, K_h, V_h)         (上の式と同じ、d_k=8 縮小版で)

# 連結
concat = [attnOut_0; attnOut_1]              shape: (T, h*d_k = d_model)
Y = concat · W_O                              shape: (T, d_model)
```

実装上は W_Q を `(h, d_model, d_k)` の 3D で持ち、head ループで分けて計算する。

### 6.7 残差接続 + LayerNorm (Tier 2)

```
# 残差 (Self-Attention 後)
X1 = X + Y                                  shape: (T, d_model)

# LayerNorm (各 token 位置 i ごとに、d_model 軸で正規化)
mu_i  = mean(X1[i, :])
var_i = var(X1[i, :])
X1_norm[i, :] = (X1[i, :] - mu_i) / sqrt(var_i + eps)
LN1_out[i, :] = gamma * X1_norm[i, :] + beta      (gamma, beta は preset 由来、shape: d_model)
```

### 6.8 FFN (Tier 2)

```
H = LN1_out · W1 + b1               shape: (T, d_ff=32)
H = GELU(H)
FFN_out = H · W2 + b2               shape: (T, d_model)

# 残差 + LayerNorm 2 回目
X2 = LN1_out + FFN_out
LN2_out = LayerNorm(X2)
```

`LN2_out` がこの Block の出力。次の Block の入力になる (Tier 3)、または最終出力 (Tier 1-2)。

### 6.9 Preset Weight の読み込み

シミュレータは起動時に **デフォルト preset** (`presets/japanese-mini-v1.json`) を読み込み、内蔵の重みとして使う。UI から別 preset に切り替えると、`createTransformer({ preset: ... })` で全パラメータが入れ替わる。

#### 6.9.1 Hand-crafted の意義 (= なぜ学習しないか)

8 サンプル × T=5 では SGD で attention を綺麗に分化させるのが難しい。Transformer は positional encoding だけで暗記できてしまうため、attention map が均一・対角線的になりかねず、教材として attention の役割を見せられない可能性が高い。

そこで本プロジェクトでは:
- **embedding を意味的に解釈可能な形に手作り** (例: dim 0 = is_noun, dim 1 = is_adjective, …)
- **W_Q / W_K を「形容詞→名詞」「指示詞→名詞」など狙った attention パターンが出るよう手作り**
- **W_V / W_O / FFN も知識参照やデモ価値のあるパターンを意図的に埋め込む**

これにより「典型的な学習結果として、こんな attention map / 知識参照が獲得される」という **理想形を確実に見せる** ことができる。Transformer の学習で似たパターンが自然に出ること自体は別途 (= 講義で論文や PyTorch demo を併用して) 補えばよい。

#### 6.9.2 Embedding の構築指針

d_model = 16 の各次元に意味を割り当てる:

| dim | 役割 |
|---|---|
| 0 | is_noun (猫, 花, 本, 小説 で 1) |
| 1 | is_adjective (大きい, 美しい, 人気 で 1) |
| 2 | is_pronoun (これ, それ, 私 で 1) |
| 3 | is_predicate (読む, 好き で 1) |
| 4 | is_は |
| 5 | is_が |
| 6 | is_を |
| 7 | is_です |
| 8 | "美しさ" 特徴 (美しい, 花 で +0.5 程度) |
| 9 | "サイズ" 特徴 (大きい, 猫 で +0.5) |
| 10 | "書物" 特徴 (本, 小説, 読む で +0.5) |
| 11 | "人気度" 特徴 (人気, 小説 で +0.5) |
| 12-15 | 予備 / token 識別の微小な one-hot 痕跡 |

これらの値はすべて `tools/build_preset.py` で素朴に書き込む。

#### 6.9.3 W_Q / W_K の構築指針

Attention の意図したパターンは「特定 dim だけが bond として機能」する形で実現する。

例: 「形容詞 (Q) → 名詞 (K)」
- `W_Q[1, 0] = 1` (adjective input → bond dim 0)
- `W_K[0, 0] = 1` (noun input → bond dim 0)
- 結果: score[adj_pos, noun_pos] = 1、それ以外 ≈ 0
- softmax 後: 形容詞は名詞だけを強く注視

例: 「指示詞 (Q) → 名詞 (K)」
- `W_Q[2, 1] = 1`
- `W_K[0, 1] = 1`

例: 「述語 (Q) → 主語名詞 (K)」 (= 動詞・形容詞動詞が直前の名詞を見る)
- `W_Q[3, 2] = 1`
- `W_K[0, 2] = 0.7`、`W_K[2, 2] = 0.3` (代名詞も少し見る)

W_V は「対応 token の意味特徴 (dim 8-15) を主に通す」よう設計し、W_O で d_model 次元に戻す。

#### 6.9.4 FFN の構築指針 (Tier 2)

W1 / b1 で「特定パターン検出ニューロン」を作る:

```
# 例: 「人気な名詞」検出ニューロン (k 番目)
W1[0, k] = +1     (is_noun)
W1[11, k] = +1    (人気度 特徴)
b1[k] = -1.5       (両方 1 のときだけ z > 0 で発火)
# → GELU 後: 「人気な名詞」位置のときだけ +、その他は ≈ 0
```

W2 でそのニューロンの活性を「trend 関連の意味特徴」に書き戻す。これで Lesson T8 で「ニューロン #k は『書物関連トークン』の処理を担当している」という knowledge memory 解釈が可能になる。

#### 6.9.5 Preset JSON フォーマット

```json
{
  "name": "japanese-mini-v1",
  "version": 1,
  "description": "Hand-crafted preset, designed to demonstrate adjective→noun and pronoun→noun attention patterns.",
  "kind": "hand-crafted",
  "config": {
    "T": 5,
    "d_model": 16,
    "h": 1,
    "d_k": 16,
    "d_ff": 32,
    "N": 1,
    "vocabSize": 16,
    "vocab": ["これ", "それ", ...]
  },
  "weights": {
    "W_E": [[...], [...], ...],     // 16x16 array
    "W_Q": [[...], [...], ...],     // 16x16
    "W_K": [...],
    "W_V": [...],
    "W_O": [...],
    "blocks": [
      {
        "W_Q": [...],
        "LN1": { "gamma": [...], "beta": [...] },
        "FFN": { "W1": [...], "b1": [...], "W2": [...], "b2": [...] },
        ...
      }
    ]
  },
  "designIntent": {
    "attentionPatterns": [
      "adjective → noun (bond dim 0)",
      "pronoun → noun (bond dim 1)",
      "predicate → noun (bond dim 2)"
    ],
    "ffnNeurons": {
      "k=12": "noun + 人気度 を検出 (= 人気な名詞)",
      "k=15": "noun + 書物 を検出 (= 書籍関連語)"
    }
  }
}
```

`trainedOn` / `trainingLoss` は削除。代わりに `kind: "hand-crafted"` と `designIntent` を持たせて、preset の意図を明示する。

#### 6.9.6 Preset 生成側 (repo 開発者向け)

`tools/build_preset.py` (Python + numpy のみ):

```python
# 概要:
# 1. 16 語彙 × 16 dim の embedding を §6.9.2 の表に従って手で書き込む
# 2. W_Q, W_K, W_V, W_O を §6.9.3 の指針で構築
# 3. (Tier 2 で) LayerNorm γ=1, β=0 / FFN W1, b1, W2, b2 を §6.9.4 で構築
# 4. JSON 形式で出力
```

これは **シミュレータ本体には含まれない** スクリプト。生成された `presets/*.json` のみが配布物に同梱される。

PyTorch 依存はゼロ。numpy だけで動くので、`uv sync` も即座に終わる。

## 7. 可視化仕様 (最重要)

MLP 版が node-edge SVG だったのに対し、**行列ヒートマップ + セル単位インタラクション** が Transformer 版の核。

### 7.1 行列ヒートマップ

各行列 (Q, K, V, X, scores, attn, ...) を **グリッドの SVG** として描画:

- 各セルは小さな矩形 (例: 28px × 28px)
- セルの色は値の符号と絶対値で決める (青=負、白=0、赤=正、明度=|値|)
- セルの中央に値 (例: `+0.412`) を 9px 程度の monospace で表示
- セルの境界は薄いグレーで区切る
- 行ヘッダに **token ID と単語ラベル** (例: `T0 これ`)、列ヘッダに次元番号 (例: `dim 0`)

### 7.2 Attention Map (特別扱い)

T × T のヒートマップで、

- 行 i = query (i 番目の token から見る)
- 列 j = key (j 番目の token に注目)
- セル値 = `attn[i, j]` (softmax 後、行ごとに合計 1)
- 色は単一色 (青) の濃淡で 0..1 を表現
- セルの中央に値 `0.42` のように表示
- **行・列ヘッダに単語ラベル** が入るので、「**これ**→**花**: 0.65」のように直接読める

これは Transformer の最も特徴的な可視化なので、補助パネルにも独立タブ「Attention Map」として再表示する (中央パネルの何タブを見ていても常駐)。

### 7.3 ホバー / クリック動作

- **ホバー**: そのセルの位置 (例: `Q[T2 美しい, 3]`) と値をツールチップで表示。同時に、計算上関連するセルをハイライト (Q[T2,3] なら X[T2,:] と W_Q[:,3] を黄色く強調)。
- **クリック**: 補助パネル「式の展開」にそのセルの計算式を 3 段で展開:
  - [段1] 一般形: `Q[i,k] = sum_d X[i,d] · W_Q[d,k]`
  - [段2] 当てはめ: `Q[T2 美しい, 3] = sum_d X[T2,d] · W_Q[d,3]`
  - [段3] 数値: `Q[T2 美しい, 3] = X[T2,0]·W_Q[0,3] + ... = +0.412`

これが MLP 版「ノードクリック → [段3] 板書」の Transformer 拡張版。

### 7.4 Pan / Zoom

中央パネルの各タブにつき、以下を提供:

- **マウスドラッグ**: 全体をパン (SVG の viewBox を更新)
- **マウスホイール**: ズーム (viewBox の width/height を変更、マウス位置中心)
- **タッチピンチ**: モバイル想定 (Tier 3 で対応)
- **「全体表示」ボタン**: viewBox を初期値にリセット
- **「100% 表示」ボタン**: ズームを 1.0 に固定

実装は SVG の `viewBox` を JS で書き換えるだけで完結する。

### 7.5 アニメーション

MLP 版の「粒が流れる」相当:

- **Forward**: 行列の計算順序に沿ってセルを順次ハイライト (X → Q → K → V → scores → attn → attnOut → Y のように、各行列の出現を 200ms ずつ遅らせて表示)
- **Attention の細部**: 1 行ずつ softmax を計算する様子を「行が右から左に塗られていく」演出

Speed スライダで `0×` (アニメなし) 〜 `3×` 切り替え。

### 7.6 FFN ニューロン解釈ビュー (Tier 2 補助機能)

FFN 中間層 (d_ff=32) の各ニューロンをクリックすると、補助パネルに「このニューロンが、このシミュレータが知っている 8 例文中のどの token に対して最も強く活性化するか」上位 3 つを表示する (= "knowledge memory" 解釈)。

例: ニューロン #15 をクリック → 「最も活性化する token: '小説' (act=2.31)、'本' (1.87)、'読む' (0.92)」 → 「この 1 ニューロンは『書物関連』を担当している可能性」と読み取れる。

実装的には、preset 学習時に各ニューロンの max-activating token を計算して preset JSON にメタ情報として書いておく (`presetMeta.ffnNeurons[neuronId] = [{ token, activation }, ...]`)。シミュレータ側は表示するだけ。

## 8. インタラクション

| 操作 | 効果 |
|---|---|
| Sentence セレクタ (8 文から選択) | 入力 token 列を変更、Forward 前の状態に戻る |
| Forward ボタン | 全行列を順次計算、アニメ付きで表示 |
| Reset ボタン | 中間結果をクリア (重みは Preset から固定) |
| Preset セレクタ | 別 preset に切り替え、行列がすべて入れ替わる |
| **セル クリック** | 補助パネル「式の展開」にその計算を表示 |
| **セル ホバー** | 関連セルをハイライト + ツールチップ |
| **パネル ドラッグ** | viewBox をパン |
| **パネル ホイール** | viewBox をズーム |
| Export (Tier 3) | 現在の重みを JSON で出力 (= preset の編集) |
| Import (Tier 3) | JSON から重みを読み込み |

Backward / Update / Run / Step 系のボタンは **存在しない** (本シミュレータは forward-only)。

## 9. 教材としての段階シナリオ (Lessons)

MLP 版の Lesson1〜8 と同じ哲学で Lesson 1〜10 を用意済み (Lesson 11 は §3.4 の検討案)。Lesson は内蔵 + UI から選択可能。すべての Lesson は **forward 動作の観察** のみで完結する。

| Lesson | 内容 | viewTab | sampleIdx | 状態 |
|---|---|---|---|---|
| 1 | Token Embedding を見る (W_E[tokens] を主役に) | embed | 0 | ✅ 実装済み |
| 2 | 位置エンコーディング (W_P) を見る | embed | 0 | ✅ 実装済み |
| 3 | Q (Query) projection を見る | qkv | 0 | ✅ 実装済み |
| 4 | K (Key) projection を見る | qkv | 0 | ✅ 実装済み |
| 5 | V (Value) projection を見る (identity の意義) | qkv | 0 | ✅ 実装済み |
| 6 | Attention Score (Q·K^T / √d_k) | attn | 0 | ✅ 実装済み |
| 7 | softmax と attention map | attn | 0 | ✅ 実装済み |
| 8 | Multi-Head — 役割分担と同時並行 | attn | 2 | ✅ 実装済み |
| 9 | 残差接続 + LayerNorm | out | 2 | ✅ 実装済み |
| 10 | FFN (検出器ニューロン) と 1 ブロック完成 | ffn | 2 | ✅ 実装済み |
| 11 | (検討案) 多ブロック化で表現が階層化する | ? | ? | 🟡 未着手 (§3.4 参照) |

各 Lesson は MLP 版同様 `hint` (1 段落 + `\n\n` 区切りの複数段落) + `checks` (3 つの観察ポイント、必要なら `\n\n💡 補足` で詳細解説) を持つ。

### 9.1 Lesson 内の数値整合性

Lesson 文中の具体的数値 (例: `scores[h0, T2 美しい, T3 花] = +3.526`、`ffn_h[T0, h13] = +1.00`) は **Python 自動テスト** (`tests/py/test_lesson_values.py`) で常時検証されている。preset を変えると Lesson 文中の値が乖離して即 CI で検知できる仕組み。これにより教材の数値正確性が長期にわたって保たれる。

## 10. 非機能要件

- **単一 HTML**: 配布物は `build/nn_sim_transformer.html` 1 ファイル。CDN 不使用。
- **サイズ目標**: **250 KB 以下** (gzip しない素の状態)。MLP 版の 174 KB に比べて、行列計算と多 Block のコードで増える分を見込むが、Backward / 学習を削ったので 350 KB → 250 KB に縮小。
- **Preset サイズ**: デフォルト preset (`japanese-mini-v1.json`) は **バンドル内に inline** で埋め込む (~20-30 KB)。追加 preset は別 JSON として配布。
- **対応ブラウザ**: Chrome / Edge / Firefox / Safari の最新 2 版。IE 不問。
- **オフライン動作**: `file://` で全機能。
- **多言語**: v1 は日本語のみ。

## 11. ファイル構成

```
nn-anatomy-transformer/
├── README.md
├── docs/
│   ├── design.md                  本書 (全機能仕様)
│   ├── math-notes-transformer.md  数式リファレンス (P8 で完備)
│   ├── lesson-plans-transformer.md  90 分授業向けプラン
│   └── implementation-notes.md    実装ログ (フェーズごと)
├── src/
│   ├── index.html                 UI エントリ
│   ├── style.css                  スタイル (行列ヒートマップ、pan/zoom)
│   └── js/
│       ├── rng.js                 (MLP 版から流用) PE 生成と内部用
│       ├── tokenizer.js           16 語固定語彙 + encode/decode
│       ├── matrix.js              flat-array ベースの行列演算ヘルパ
│       ├── presets.js             バンドル内 inline preset (デフォルト)
│       ├── model.js               createTransformer / forward
│       ├── lessons.js             Lesson T1〜T9 定義
│       ├── view.js                行列ヒートマップ描画 + pan/zoom
│       ├── explain.js             「式の展開」HTML テンプレート
│       └── controller.js          UI とモデルの接続
├── presets/                       (配布用 JSON、開発者が train_preset.py で生成)
│   ├── japanese-mini-v1.json      デフォルト (8 文学習済)
│   ├── induction-head.json        induction head 風の attention
│   ├── knowledge-lookup.json      FFN が知識参照風に動く
│   └── random-init.json           ランダム初期化 (比較用)
├── tests/
│   ├── js/                        node:test 単体テスト
│   ├── py/                        PyTorch 参照実装 + フィクスチャ生成
│   └── fixtures/
├── tools/
│   ├── bundle.py                  (MLP 版から流用) 単一 HTML 生成
│   ├── build_preset.py            §6.9 の指針に従って hand-crafted preset JSON を構築 (numpy のみ)
│   └── inline_preset.py           デフォルト preset を src/js/presets.js に展開
├── build/
│   └── nn_sim_transformer.html
├── Makefile
├── pyproject.toml                 numpy + pytest (PyTorch 依存なし)
├── package.json
└── .gitignore
```

MLP 版 v1 から **そのまま流用**: `rng.js` (内部の決定的乱数のみ、学習には使わない), `tools/bundle.py`, `Makefile` の骨組み, 設計書スタイル, テスト方針。

## 12. 実装ロードマップ

| フェーズ | 内容 | 状態 |
|---|---|---|
| P0 | 設計書 (本書) + プロジェクト雛形 | ✅ 完了 |
| P1 | matrix.js + tokenizer.js + 数値参照実装 (numpy) | ✅ 完了 (matmul / softmax / layernorm / gelu が numpy と 1e-12 で一致) |
| P2 | tools/build_preset.py + デフォルト preset 生成 | ✅ 完了 (japanese-mini-v1.json、§6.9 の指針 + Multi-Head 拡張) |
| P3 | Tier 1 model.js (forward only) + presets.js inline | ✅ 完了 (Tier 2 で再構築済み) |
| P4 | Tier 1 view.js (静的描画) + pan/zoom | ✅ 完了 |
| P5 | Tier 1 explain.js + interactivity | ✅ 完了 |
| P6 | Lesson 1-5 (Tier 1 範囲、Q/K/V を 3 つに分割した経緯あり) + 8 文セレクタ | ✅ 完了 |
| P7a | Multi-Head Attention (h=2, d_k=8) + Lesson 8 | ✅ 完了 |
| P7b | 残差接続 + LayerNorm + Lesson 9 | ✅ 完了 |
| P7c | FFN (2 層 MLP, GELU) + Lesson 10 | ✅ 完了 |
| **P8** | **多ブロック化 (案) + 複数 Preset 切替 + Lesson 11 + ドキュメント完備** | 🟡 未着手 (§3.4 参照) |

**v0.4 時点で P0〜P7c 完了**、Lesson 1〜10 で 1 ブロック分の forward 全工程を観察可能。

各フェーズの末尾に「`make test` 全パス」「numpy 参照と数値一致 (1e-12 オーダ)」「手動チェックリスト消化」を必須とする。

各フェーズの末尾に「`make test` 全パス」「PyTorch 参照と数値一致 (1e-10 オーダ)」「手動チェックリスト消化」を必須とする。

(MLP 版の P0-P12 比で約 4 フェーズ分短縮。Backward / 学習を削った分。)

## 13. 検証方針

### 13.1 数値一致テスト (核)

MLP 版と同様、Transformer 版も **「JS Transformer forward = numpy 参照 forward」** を 1e-12 オーダで一致させる:

- `tests/py/reference.py` で numpy を使って参照 forward を計算
- 同じ重み (preset の JSON) と同じ入力 token 列で fixture (`tests/fixtures/*.json`) を生成
- JS 側でも同じ計算をして、各中間結果 (Q, K, V, scores, attn, ...) を fixture と数値比較

### 13.2 NumPy のみ依存 (PyTorch 不要)

学習ループを持たない (= preset は hand-crafted) ため、autograd が必要なく numpy で十分。LayerNorm や softmax の式は `tests/py/reference.py` に明示的に書いておくため、PyTorch との数値合わせも不要。

将来 v2 で Backward + 学習可視化を追加する際には PyTorch を再投入する想定 (= 拡張余地に残す)。

### 13.3 単体テスト

`tests/js/*.test.mjs`:
- `matrix.js` の matmul, softmax, layernorm, gelu の数値テスト
- `tokenizer.js` の encode/decode 整合性
- `model.js` の forward 入出力形状チェック + PyTorch fixture 一致
- `lessons.js` の Lesson 定義整合性
- preset JSON の構造バリデーション

`tests/py/*.py`:
- 参照実装が想定どおり動くこと
- preset 学習スクリプトが想定の loss まで収束する
- バンドル健全性

### 13.4 手動チェックリスト

各フェーズに「人間が画面を見て確認すべき項目」を 3-5 項目用意。

## 14. 拡張余地 (v2 以降)

本シミュレータでスコープ外としたが、将来追加候補:

- **Backward + 学習可視化**: 当初構想に含めていたが、MLP 版で chain rule を学習済みのユーザーには不要と判断 → v2 として、softmax の Jacobian と 残差を経由する勾配の和を可視化する派生 Lesson を追加可能。このとき PyTorch 依存を再投入する想定。
- **学習済みモデルからの preset 抽出**: BERT / GPT-2 等の小型モデルから 1 head 分を抜いて preset 化、現状の hand-crafted preset と比較できる "real_model_extracted" preset を追加。Lesson T9 にこれを加えると「hand-crafted で見せた典型例が、本物の学習結果でも (近似的に) 出ている」と橋渡しできる
- **Causal Mask の可視化**: GPT 系 (decoder only) に必要。Self-Attention に上三角を −∞ にする処理を可視化
- **Encoder-Decoder 構造**: Seq2Seq、機械翻訳の基礎
- **より長い sequence (T=10-20)**: pan/zoom で耐えられる範囲で
- **モデル比較ビュー**: 同じ入力に対し 2 preset の attention map を並べて差分可視化
- **多言語対応**: en + 中国語等。vocab を切り替える機構
- **言語モデル inference**: 学習済 preset を使った next-token 予測の一連動作 (forward → softmax → argmax)
- **WebGPU / WebGL によるバッチ計算高速化**: T を大きくしたい場合
- **Preset エディタ**: シミュレータ内で重みを直接編集して attention 変化を観察 (= MLP 版 B3 の Transformer 版)

## 15. 開発環境・セットアップ

### 必要ソフト

- **uv 0.11+** (Python 環境管理)
- **Python 3.10+** (テスト用、preset 構築用)
- **Node.js 20+** (JS 単体テスト用、`node --test`)
- **モダンブラウザ** (動作確認用)

### 初回セットアップ

```bash
git clone <repo>
cd nn-anatomy-transformer
uv sync          # Python 依存 (numpy + pytest) を .venv に入れる (数秒)
node --version   # v20 以上を確認
make help        # 利用可能なタスク一覧
```

### 開発タスク

```bash
make serve         # ローカルサーバ (http://localhost:8000/)
make test          # JS + Python 全テスト
make test-js       # JS 単体テスト
make test-py       # Python (参照 + バンドル)
make fixtures      # numpy 参照から数値検証 fixture を再生成
make build-preset  # tools/build_preset.py を実行、presets/japanese-mini-v1.json を再生成
make bundle        # → build/nn_sim_transformer.html
```

## 16. リスク・制約

| リスク | 影響 | 対策 |
|---|---|---|
| 行列が大きすぎて画面に収まらない | 教育効果低下 | pan/zoom 必須、Tier 1 では T=5 d_model=16 に固定 |
| Hand-crafted の「素朴さ」を学生が「本物っぽくない」と感じる | 信頼性低下 | Lesson T1 で透明に「これは hand-crafted のデモ用重み。学習で似たパターンが獲得される」と明示。実際の Transformer の attention map (BERT 等) との対比図も提示 |
| Hand-crafted のクオリティが低いと教材として薄い | 体験低下 | preset 設計を §6.9 で表にまとめ、各 W_Q[a, b] = 1 の意図を明記。Lesson T2-T4 で「なぜこういう attention が出るか」を W_Q / W_K の設計と紐づけて説明 |
| FFN ニューロン解釈が hand-crafted で「あらかじめ仕込んである」感が強い | わざとらしさ | 「k=12 のニューロンに『人気な名詞』検出を仕込んだ」と素直に書く。「本物の学習でもこういうニューロンが自発的に獲得される (Geva 2021 等の論文を参照)」と論文への橋渡しで補完 |
| バンドルサイズ 250 KB を超過 (preset inline 込み) | 配布性低下 | デフォルト preset の重み精度を落とす (float64 → float32 で半分)、追加 preset は inline せず別 JSON ファイルに分離 |
| MLP 版の「ノード」概念から「行列セル」概念への切り替えで混乱 | 学習者離脱 | Lesson T1 で「これが Transformer の見方」と明示的に橋渡しする教材を入れる |
| Preset を変えても見た目が大きく変わらない (= attention が単調) | 比較 Lesson が成立しない | preset 設計時にわざと特徴的な attention pattern を作る (induction head 用、`random-init` 等)、`random-init` preset との対比で最低限の差分を保証 |

---

## 付録 A. 用語表

- **Token**: 入力系列の 1 単位。本プロジェクトでは固定 16 語 (日本語) のいずれか。
- **d_model**: 埋め込み次元、ブロック間で一定。Tier 1 では 16。
- **d_k**: 1 head の Q/K/V の次元 = `d_model / h`。Tier 1 では 16 (h=1)、Tier 2 では 8 (h=2)。
- **d_ff**: FFN 中間層の次元。Tier 2 では 32。
- **Q, K, V**: Query, Key, Value。Attention 機構の 3 役。
- **Self-Attention**: Q と K と V が **すべて同じ X から作られる** 場合の Attention。
- **Multi-Head**: h 個の Self-Attention を並列に動かし、結果を連結する。
- **Causal Mask**: 未来の token を見ないように attention score の上三角を −∞ にする工夫。GPT 系で使用 (本シミュレータでは v2 拡張)。
- **LayerNorm**: 各 token 位置 (= d_model 軸) で平均 0 / 分散 1 に正規化。
- **GELU**: Gaussian Error Linear Unit。ReLU の滑らかな代替。
- **Preset**: 事前学習済みの重みセット。シミュレータは preset の重みを使って forward だけを行う。
- **Knowledge memory (FFN 解釈)**: FFN の中間層ニューロンが特定 token / 特定意味に反応する現象を「ニューロン単位の知識ストア」と見立てる解釈 (Geva et al. 2021 系)。

## 付録 B. 設計上の決定事項一覧

1. **forward 専用**: backward / 学習はシミュレータ側で扱わない。MLP 版で chain rule を習得済みであることが前提。
2. **Hand-crafted preset**: 重みは Python (numpy) で hand-design して JSON 化、シミュレータは読み込むだけ。SGD で学習させない (= 8 サンプルでは attention が綺麗に分化しないリスクが高いため)。
3. **NumPy のみ依存**: PyTorch を入れない。autograd が要らないので。
4. **日本語 16 語固定 vocab + T=5**: attention map に意味を持たせるため。
5. **flat-array layout**: 行列を Float64Array の 1 次元で持つ。
6. **行列ヒートマップ中心**: node-edge 図は使わない。各セルをクリック可能にして MLP 版の哲学を継承。
7. **pan/zoom 必須**: 行列が画面に収まらないので最初から組み込む。viewBox 操作で完結。
8. **causal mask は v2 から**: Tier 1-3 では bidirectional な attention に絞る (BERT 寄り)。
9. **FFN ニューロン解釈ビューを目玉機能に**: knowledge memory 解釈を hand-crafted で確実に見せる。Attention だけでなく FFN の役割も実感できる。
10. **Lesson curriculum を MLP 版 v1 のフォーマットで**: hint + checks(3) + 補足。Lesson T1 冒頭で「重みは hand-crafted」と透明に明示する。

## 付録 C. デフォルト Preset の対象例文 (再掲)

`tools/build_preset.py` で `japanese-mini-v1.json` を構築する際に「これらの文で意図した attention map / FFN 活性が出るように」設計する対象 8 文:

```
これ は 美しい 花 です       ← 「美しい→花」「これ→花」
それ は 人気 小説 です       ← 「人気→小説」「それ→小説」
私 は 猫 が 好き             ← 「私→好き」「猫→好き」
私 は 本 を 読む             ← 「私→読む」「本→読む」
私 は 小説 を 読む           ← (4) と同パターン、目的語比較
大きい 猫 は 美しい です     ← 「大きい→猫」「美しい→猫」
美しい 花 は 人気 です       ← 「美しい→花」「人気→花」
これ は 大きい 本 です       ← 「大きい→本」「これ→本」
```

これらの「→」関係が softmax 後の attention 0.5 以上で実現するよう、§6.9.3 の指針に従って W_Q / W_K を構築する。学習はしない。

---

(以上、設計書 v0.2。実装着手時に必要に応じて改訂)
