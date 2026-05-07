# 数学ノート — Transformer 版

> シミュレータが内部で実行している計算を、数式と導出のレベルで整理したリファレンス。Lesson の hint や checks では「観察として何が見えるか」を中心に書いてあるが、本ノートは「なぜそういう式なのか」「どうやって計算しているか」をまとめる。
>
> 関連: [`design.md`](design.md) §6 計算仕様 (実装寄り)、Lesson 1〜10 (画面で見ながらの観察)。

---

## 0. 表記と寸法

本シミュレータの主要パラメータ:

| 記号 | 値 | 意味 |
|---|---|---|
| `T` | 5 | 入力トークン数 (固定長) |
| `d_model` | 16 | 埋め込み次元 |
| `h` | 2 | Multi-Head の head 数 |
| `d_k` | 8 | 1 head あたりの Q/K/V 次元 (`d_model / h`) |
| `d_ff` | 32 | FFN 中間層の次元 |
| `vocabSize` | 16 | 固定語彙サイズ |
| `ε` | 1e-5 | LayerNorm の数値安定化定数 |

行列・ベクトルは原則 **行ベクトル並び**: 入力 `X ∈ ℝ^(T × d_model)` の各行 `X[t, :]` が 1 トークンの埋め込み。索引は 0 始まり。

### 添え字の記法

- `t, i, j ∈ {0, ..., T-1}`: トークン位置
- `d ∈ {0, ..., d_model-1}`: 埋め込み dim
- `k ∈ {0, ..., d_k-1}`: head 内の Q/K/V dim
- `hi ∈ {0, ..., h-1}`: head 番号
- `g ∈ {0, ..., h·d_k-1}`: Q_full / K_full / V_full の global col 番号 (`g = hi·d_k + k`)

---

## 1. Token Embedding と位置エンコーディング

入力は token id 列 `tokens ∈ {0, ..., vocabSize-1}^T`。これを実数ベクトル `X ∈ ℝ^(T × d_model)` に変換する。

```
X[t, d] = W_E[tokens[t], d] + W_P[t, d]
```

### 1.1 W_E (埋め込み行列)

`W_E ∈ ℝ^(vocabSize × d_model)`。本 preset では各 token に対して **固定の構造化値** を hand-crafted で書き込んでいる:

- 1 次カテゴリ (d0..d7): 1 つの dim だけ +3.0、他は 0。例: 名詞 token は d0 = +3、形容詞 token は d1 = +3
- 2 次特徴 (d8..d11): 2〜3 token に共通する意味属性、+1.0。例: 「美しい」「花」は共に d8 = +1.0 (美しさ特徴)
- 識別痕跡 (d12..d15): `token_id mod 4` で 1 つの dim に +0.3 (token 単独識別用の微小信号)

### 1.2 W_P (位置エンコーディング)

`W_P ∈ ℝ^(T × d_model)`、Vaswani et al. 2017 の sin/cos 方式:

```
W_P[t, 2i]   = sin(t / 10000^(2i / d_model))
W_P[t, 2i+1] = cos(t / 10000^(2i / d_model))
```

- 偶数 dim は sin、奇数 dim は cos
- 分母 `10000^(2i/d_model)` は `i` が大きいほど大きい (= 周期が長い)
- `i = d/2`、つまり dim 0/1 は周期 2π、dim 14/15 は周期 ≈ 2π · 5623

### 1.3 周波数階層の効果

低 dim (d0..d3) は **位置に敏感** (sin/cos が ±1 の幅で変動)、高 dim (d12..d15) は **位置にほぼ不感** (cos(0 に近い値) ≈ 1 の定数)。これにより同じ token が違う位置に出ても **重複しない指紋** を持つ。

線形写像で「位置 t と t+k の関係」を取り出せる性質が知られている (Vaswani et al. 2017 §3.5)。本シミュレータでは深く扱わない。

---

## 2. Q / K / V 射影

```
Q_full = X · W_Q   shape: (T, h·d_k)
K_full = X · W_K   shape: (T, h·d_k)
V_full = X · W_V   shape: (T, h·d_k)
```

`W_Q, W_K, W_V ∈ ℝ^(d_model × h·d_k)`。本シミュレータでは `h · d_k = d_model = 16` と取っているので、shape は `(16, 16)`。

各成分は明示的に:

```
Q_full[t, g] = Σ_d X[t, d] · W_Q[d, g]
```

### 2.1 Multi-Head 分割

`Q_full` を 3 階テンソル `Q ∈ ℝ^(T × h × d_k)` にリシェイプして head 別に扱う:

```
Q[t, hi, k] = Q_full[t, hi · d_k + k]
```

K, V も同様。本シミュレータの内部メモリでは `Q_full` と `Q` は **同じバイト列** で、stride で head 軸を切り出している。

### 2.2 bond の概念 (preset の hand-crafted 構造)

「W_Q[d_in, g]」を「入力 dim d_in の値を出力 col g に流す重み」と読む。本 preset では 3 つの非ゼロエントリしかない:

| 名前 | W_Q | W_K | 意図 |
|---|---|---|---|
| bond 0 | `W_Q[d1, g=0] = 1` | `W_K[d0, g=0] = 1` | 形容詞 (Q) → 名詞 (K)、head 0 col 0 |
| bond 1 | `W_Q[d2, g=1] = 1` | `W_K[d0, g=1] = 1` | 指示詞 (Q) → 名詞 (K)、head 0 col 1 |
| bond 2 | `W_Q[d3, g=8] = 1` | `W_K[d0, g=8] = 1` | 述語   (Q) → 名詞 (K)、head 1 col 0 |

`W_V = I` (単位行列) なので `V = X` となる。教育用の選択。

---

## 3. Scaled Dot-Product Attention

各 head 独立に:

```
scores[hi, i, j] = (Q[i, hi, :] · K[j, hi, :]) / √d_k
                 = (Σ_k Q[i, hi, k] · K[j, hi, k]) / √d_k

attn[hi, i, :]   = softmax(scores[hi, i, :])    (= 各行の和 = 1)

out[t, hi, :]    = Σ_j attn[hi, t, j] · V[j, hi, :]
```

### 3.1 √d_k で割る理由

`Q[i, hi, :]` と `K[j, hi, :]` の各成分が独立に平均 0・分散 1 の確率変数だと仮定すると、内積の分散は `d_k` になる。`d_k` が大きいと scores の分散も大きくなり、softmax が極端に振り切る (1 つだけ ≈ 1、他はすべて ≈ 0)。これを抑制するため `√d_k` で割って分散を 1 程度に正規化する。

本シミュレータでは `d_k = 8` なので除数は `√8 ≈ 2.828`。

### 3.2 softmax (数値安定形)

```
softmax(z)[j] = exp(z[j] - max(z)) / Σ_j' exp(z[j'] - max(z))
```

最大値を引く操作 (`z - max(z)`) は softmax の値を変えないが、`exp` 入力を ≤ 0 に揃えてオーバーフローを防ぐ。本シミュレータの実装でも採用 (`tools/build_preset.py` の `_softmax`、`src/js/matrix.js` の `softmax`)。

### 3.3 attention の意味

`attn[hi, i, j]` = 「head hi で、トークン i がトークン j にどれだけ注目するか」の確率。各行 `attn[hi, i, :]` が和 1 の確率分布になる。

`out[t, hi, :]` は「head hi で t が注目した重み付き和」で、`V` (= X、本 preset では identity 通過) を attention で集めたもの。

---

## 4. Multi-Head の連結と W_O

各 head の出力 `out ∈ ℝ^(T × h × d_k)` を head 軸方向で **concat** して `(T, h·d_k = d_model)` にし、最終射影 `W_O ∈ ℝ^(d_model × d_model)` を掛ける:

```
attnOut_concat[t, g] = out[t, g / d_k, g mod d_k]    (= out のメモリ並び替えなしのコピー)

Y[t, d] = Σ_g attnOut_concat[t, g] · W_O[g, d]
```

本 preset では `W_O = I` なので `Y = attnOut_concat`。各 head の出力が連結されて並ぶだけ。

---

## 5. 残差接続 + LayerNorm 1 (attention 部の後処理)

```
residual1[t, d] = X[t, d] + Y[t, d]
ln1_out[t, d]   = LayerNorm(residual1)[t, d]
```

### 5.1 残差接続の意義

`Y` は「attention で集めた他トークンの情報」だけ。元の token 自身の埋め込み `X` を加えることで「**自分** + **集めた情報**」の合算ベクトルにする。これが無いと attention で薄まった「自分のアイデンティティ」が深い層で消失する。

加えて、深い Transformer (層を多数積む) では勾配消失の防止にも効くが、本シミュレータは forward のみのため backward 議論は省略 (姉妹 nn-anatomy 参照)。

### 5.2 LayerNorm (per-token、d_model 軸方向)

各 token (= 各行) について独立に:

```
μ_t = (1/d_model) · Σ_d residual1[t, d]
σ²_t = (1/d_model) · Σ_d (residual1[t, d] - μ_t)²
ln1_out[t, d] = γ[d] · (residual1[t, d] - μ_t) / √(σ²_t + ε) + β[d]
```

PyTorch の `F.layer_norm` と互換。**有偏分散** (分母 `d_model`、ddof=0) を使う。

### 5.3 γ, β の役目と本 preset の選択

`γ ∈ ℝ^(d_model)` (スケール)、`β ∈ ℝ^(d_model)` (シフト) は学習可能パラメータ。本 preset では教育用に **γ = 1, β = 0** で固定。これにより LN は「純粋な平均 0・標準偏差 1 への標準化」だけを行う。

実 Transformer では γ, β も学習で「適切な dim 別スケール」を獲得する。

---

## 6. FFN (Feed-Forward Network)

per-token に作用する 2 層 MLP:

```
ffn_pre[t, k] = Σ_d ln1_out[t, d] · W1[d, k] + b1[k]
ffn_h[t, k]   = GELU(ffn_pre[t, k])
ffn_out[t, d] = Σ_k ffn_h[t, k] · W2[k, d] + b2[d]
```

形状: `W1 ∈ ℝ^(d_model × d_ff)`, `W2 ∈ ℝ^(d_ff × d_model)`, `b1 ∈ ℝ^(d_ff)`, `b2 ∈ ℝ^(d_model)`。

### 6.1 per-token 性

行列積は token 軸方向に独立 — つまり `ffn_h[t, :]` の計算には `ln1_out[t, :]` だけしか使わず、他のトークンの情報は使わない。**attention は token 間で混ぜる、FFN はトークン単位で変換** という役割分担。

### 6.2 GELU 活性化関数

Gaussian Error Linear Unit。tanh 近似版を採用 (Vaswani et al. 2017 / Hendrycks & Gimpel 2016 と整合):

```
GELU(x) = 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
```

性質: 入力 `x` が大きく正なら `≈ x`、負なら `≈ 0` を返す滑らかな関数。ReLU と違い x=0 周辺で微分可能。

### 6.3 本 preset の検出器ニューロン設計

`W1`, `b1` を hand-crafted で「特定 dim の同時発火検出器」として作ってある:

| ニューロン | W1 配線 | b1 | 意図 |
|---|---|---|---|
| h0..h11 | `W1[di, hi] = +1.0` (1 つだけ) | -1.0 | 単 dim 検出器 (ln1_out[di] が +1.0 を超えたら発火) |
| h12 | `W1[d0, 12] = +0.6, W1[d8, 12] = +0.6` | -0.5 | AND: 名詞 ∧ 美しさ |
| h13 | `W1[d0, 13] = +0.6, W1[d9, 13] = +0.6` | -0.5 | AND: 名詞 ∧ サイズ |
| h14 | `W1[d0, 14] = +0.6, W1[d10, 14] = +0.6` | -0.5 | AND: 名詞 ∧ 書物 |
| h15 | `W1[d0, 15] = +0.6, W1[d11, 15] = +0.6` | -0.5 | AND: 名詞 ∧ 人気度 |
| h16..h31 | 全 0 | 0 | 未使用 |

### 6.4 W2 による「書き戻し」

`W2` で h0..h11 は identity (h_i → 出力 d_i) で feature 維持、h12..h15 は対応する特徴 dim に +0.5 を追加加算 (= 「美しい名詞」を検出したら美しさ特徴を強化)。

これが Geva et al. 2021 系の **「FFN は knowledge memory として動く」** という解釈の最小実装。

---

## 7. 残差接続 + LayerNorm 2 (block 最終出力)

FFN 後に再度残差 + LN:

```
residual2[t, d] = ln1_out[t, d] + ffn_out[t, d]
ln2_out[t, d]   = LayerNorm(residual2)[t, d]
```

`ln2_out` が 1 ブロック分の最終出力。本シミュレータは `N = 1` なのでこれが block 出力 = ネットワーク出力。多ブロック (Tier 3 / Lesson 11 検討案) では `ln2_out` が次のブロックの `X` になる。

---

## 8. Block 全体の流れ (まとめ)

```
X (T × d_model)
  │
  ├── Q = X · W_Q,  K = X · W_K,  V = X · W_V    (T × h · d_k)
  │     │
  │     ├── per head:
  │     │     scores = (Q · Kᵀ) / √d_k
  │     │     attn   = softmax(scores)
  │     │     out    = attn · V                  (T × d_k)
  │     │
  │     └── concat heads → · W_O → Y             (T × d_model)
  │
  ├── residual1 = X + Y
  ├── ln1_out   = LayerNorm(residual1)            ← attention 部 完了
  │
  ├── FFN: ffn_h = GELU(ln1_out · W1 + b1)
  │        ffn_out = ffn_h · W2 + b2
  │
  ├── residual2 = ln1_out + ffn_out
  └── ln2_out   = LayerNorm(residual2)            ← block 出力
```

各段の shape 推移: `(T, d_model) → (T, h·d_k) → (T, h·d_k) → (T, d_model) → (T, d_model) → (T, d_ff) → (T, d_model) → (T, d_model)`。最終的に `T × d_model` に戻ってくる。

---

## 9. 数値の正確性

JS forward (`src/js/model.js`) と numpy 参照実装 (`tools/build_preset.py` の `forward_attention`) は **1e-12 オーダーで一致** することが `tests/js/test_model.test.mjs` の `forward: 8 例文すべての中間結果が numpy 参照と 1e-12 で一致` で常時検証されている。

加えて、Lesson 文中に書いた具体的な数値 (例: `scores[h0, T2 美しい, T3 花] = +3.526`、`ffn_h[T0, h13] = +1.00`) は `tests/py/test_lesson_values.py` で固定値として assert されており、preset を変えると Lesson 文が古くなることが CI で即検知される。

---

## 10. 参考文献

- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS. — 元論文。Self-Attention、Multi-Head、PE の sin/cos 方式の出典
- Hendrycks, D., & Gimpel, K. (2016). *Gaussian Error Linear Units (GELUs)*. — GELU と tanh 近似の出典
- Geva, M., Schuster, R., Berant, J., & Levy, O. (2021). *Transformer Feed-Forward Layers Are Key-Value Memories*. EMNLP. — FFN を knowledge memory と見る解釈の根拠
- Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). *Layer Normalization*. — LN の原典
