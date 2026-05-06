// model.js — Transformer Self-Attention forward (Tier 2: Multi-Head 対応)。
//
// 設計書 §5.1, §6 + §6.6 (Multi-Head) に対応。preset (hand-crafted の JSON) を
// 受け取って、入力 token 列に対する forward を計算する。学習機能は無い。
//
// すべての行列は Float64Array の 1 次元配列で持つ (matrix.js の規約)。
// アクセスは A[r * cols + c]。Multi-Head の Q/K/V/scores/attn/attnOut も
// 同じ Float64Array に格納し、head 軸はストライドで暗黙化する:
//   Q.length = T * h * d_k
//   Q[t, hi, k] = Q[(t * h + hi) * d_k + k]
//   scores.length = h * T * T
//   scores[hi, i, j] = scores[hi * T * T + i * T + j]

import {
  matmul,
  softmax,
  gelu,
  fromNestedArray,
} from "./matrix.js";

/**
 * preset (= JSON object) から Transformer net インスタンスを構築する。
 *
 * @param {object} args.preset  presets/japanese-mini-v1.json と同じ形式の object
 * @returns {object} net — 重みと中間結果バッファをすべて持つ object
 */
export function createTransformer({ preset }) {
  if (!preset || !preset.config || !preset.weights) {
    throw new Error("createTransformer: preset には config と weights が必要");
  }
  const { T, d_model, h, d_k, vocabSize, N } = preset.config;
  if (h * d_k !== d_model) {
    throw new Error(`createTransformer: h*d_k (${h * d_k}) != d_model (${d_model})`);
  }
  if (N !== 1) {
    throw new Error(`Tier 2 の model.js は N=1 のみ対応。preset N=${N}`);
  }

  // 重み (固定、preset から読み込み)
  const W_E = fromNestedArray(preset.weights.W_E);     // (vocabSize, d_model)
  const W_P = fromNestedArray(preset.weights.W_P);     // (T, d_model)
  const block0 = preset.weights.blocks[0];
  const W_Q = fromNestedArray(block0.W_Q);             // (d_model, h*d_k)
  const W_K = fromNestedArray(block0.W_K);             // (d_model, h*d_k)
  const W_V = fromNestedArray(block0.W_V);             // (d_model, h*d_k)
  const W_O = fromNestedArray(block0.W_O);             // (h*d_k, d_model)
  // LayerNorm パラメータ (γ=1, β=0 で固定。学習要素ゼロ)
  // 1D ベクトルなので fromNestedArray (2D 用) は使わず Float64Array.from で読み込む。
  const LN1_gamma = Float64Array.from(block0.LN1_gamma); // (d_model,)
  const LN1_beta  = Float64Array.from(block0.LN1_beta);  // (d_model,)
  // FFN: 2 層 MLP (d_model → d_ff → d_model) + GELU
  const d_ff = preset.config.d_ff;
  const FFN_W1 = fromNestedArray(block0.FFN_W1);         // (d_model, d_ff)
  const FFN_b1 = Float64Array.from(block0.FFN_b1);       // (d_ff,)
  const FFN_W2 = fromNestedArray(block0.FFN_W2);         // (d_ff, d_model)
  const FFN_b2 = Float64Array.from(block0.FFN_b2);       // (d_model,)
  // FFN 後の 2 回目残差 + LN2
  const LN2_gamma = Float64Array.from(block0.LN2_gamma); // (d_model,)
  const LN2_beta  = Float64Array.from(block0.LN2_beta);  // (d_model,)

  // 中間結果バッファ (forward で書き込まれる)
  const tokens   = new Int32Array(T);
  const X        = new Float64Array(T * d_model);
  // Q/K/V は (T, h, d_k) を 1D 化。サイズは T * h * d_k = T * d_model
  const Q        = new Float64Array(T * h * d_k);
  const K        = new Float64Array(T * h * d_k);
  const V        = new Float64Array(T * h * d_k);
  // scores / attn は (h, T, T) を 1D 化。サイズは h * T * T
  const scores   = new Float64Array(h * T * T);
  const attn     = new Float64Array(h * T * T);
  // attnOut は (T, h, d_k) (= (T, h*d_k) と同等のメモリ)
  const attnOut  = new Float64Array(T * h * d_k);
  const Y        = new Float64Array(T * d_model);
  // 残差 + LayerNorm (Lesson 9)
  const residual1 = new Float64Array(T * d_model);
  const ln1_out   = new Float64Array(T * d_model);
  // FFN (Lesson 10)
  const ffn_pre  = new Float64Array(T * d_ff);   // GELU 入力 (= ln1_out · W1 + b1)
  const ffn_h    = new Float64Array(T * d_ff);   // GELU 出力 (= 検出器の活性)
  const ffn_out  = new Float64Array(T * d_model); // = ffn_h · W2 + b2
  const residual2 = new Float64Array(T * d_model); // = ln1_out + ffn_out
  const ln2_out   = new Float64Array(T * d_model); // block 最終出力

  return {
    // ハイパパラメータ
    T, d_model, h, d_k, d_ff, vocabSize, N,
    vocab: preset.config.vocab.slice(),

    // 重み (固定)
    W_E, W_P,
    blocks: [{
      W_Q, W_K, W_V, W_O,
      LN1_gamma, LN1_beta,
      FFN_W1, FFN_b1, FFN_W2, FFN_b2,
      LN2_gamma, LN2_beta,
    }],

    // 中間結果
    tokens, X, Q, K, V, scores, attn, attnOut, Y,
    residual1, ln1_out,
    ffn_pre, ffn_h, ffn_out, residual2, ln2_out,

    // メタ
    presetName: preset.name ?? "(unknown)",
    presetMeta: {
      kind: preset.kind,
      version: preset.version,
      tier: preset.tier,
      description: preset.description,
      designIntent: preset.designIntent,
    },
    phase: "idle",
  };
}

/**
 * Tier 2 forward (Multi-Head Self-Attention):
 *   X = W_E[tokens] + W_P
 *   Q_full = X · W_Q,  K_full = X · W_K,  V_full = X · W_V    (T × h*d_k)
 *   per head hi:
 *     scores_hi[i, j] = (Q[t=i, hi, :] · K[t=j, hi, :]) / √d_k
 *     attn_hi   = softmax(scores_hi)
 *     out_hi    = attn_hi · V_hi    (T × d_k)
 *   concat heads → (T, h*d_k = d_model)
 *   Y = concat · W_O     (T × d_model)
 *
 * 入力 tokens は長さ T の Int32Array (vocab id 列)。net.tokens にコピーして
 * net.X / Q / K / V / scores / attn / attnOut / Y を順に埋める。
 *
 * @param {object} net
 * @param {Int32Array | number[]} tokens
 * @returns {Float64Array} net.Y (出力)
 */
export function forward(net, tokens) {
  const { T, d_model, h, d_k, vocabSize } = net;
  if (tokens.length !== T) {
    throw new Error(`forward: tokens.length=${tokens.length} !== T=${T}`);
  }
  // tokens を内部にコピー、範囲チェック
  for (let t = 0; t < T; t++) {
    const id = tokens[t];
    if (!Number.isInteger(id) || id < 0 || id >= vocabSize) {
      throw new Error(`forward: tokens[${t}]=${id} は範囲外 (0..${vocabSize - 1})`);
    }
    net.tokens[t] = id;
  }

  // (1) X = W_E[tokens] + W_P
  for (let t = 0; t < T; t++) {
    const tokenId = net.tokens[t];
    const wOff = tokenId * d_model;
    const pOff = t * d_model;
    const xOff = t * d_model;
    for (let d = 0; d < d_model; d++) {
      net.X[xOff + d] = net.W_E[wOff + d] + net.W_P[pOff + d];
    }
  }

  // (2) Q/K/V を「concat 全 head」の形で計算: shape (T, h*d_k) = (T, d_model)
  // matmul の出力サイズに合わせて、Q/K/V も内部レイアウトでは (T, h*d_k) = (T, d_model) と等価。
  const block0 = net.blocks[0];
  const hd = h * d_k;
  matmul(net.X, T, d_model, block0.W_Q, hd, net.Q);
  matmul(net.X, T, d_model, block0.W_K, hd, net.K);
  matmul(net.X, T, d_model, block0.W_V, hd, net.V);

  // (3) head ごとの scores/attn/out を計算
  // Q[t, hi, k] = net.Q[(t * h + hi) * d_k + k]?  ← W_Q の cols 0..hd-1 は
  //   col index = hi * d_k + k に対応する (hi: head index, k: head 内の dim)。
  // つまり Q の格納順は (T, h, d_k) を [t][hi][k] フラット化したものではなく、
  // (T, h*d_k) を [t][hi*d_k + k] とした **同じ** メモリレイアウト。
  // → どちらの解釈でも net.Q[(t * h + hi) * d_k + k] = net.Q[t * hd + hi * d_k + k] と同値。
  const invSqrtDk = 1 / Math.sqrt(d_k);
  // scores buffer は (h, T, T)
  for (let hi = 0; hi < h; hi++) {
    const sOff = hi * T * T;
    for (let i = 0; i < T; i++) {
      const qOff = i * hd + hi * d_k;
      for (let j = 0; j < T; j++) {
        const kOff = j * hd + hi * d_k;
        let s = 0;
        for (let k = 0; k < d_k; k++) {
          s += net.Q[qOff + k] * net.K[kOff + k];
        }
        net.scores[sOff + i * T + j] = s * invSqrtDk;
      }
    }
    // softmax row-wise on (T, T) sub-buffer
    softmax(net.scores.subarray(sOff, sOff + T * T), T, T,
            net.attn.subarray(sOff, sOff + T * T));
    // attnOut[t, hi, :] = sum_j attn[hi, t, j] * V[j, hi, :]  → (T, d_k)
    for (let i = 0; i < T; i++) {
      const oOff = i * hd + hi * d_k;
      // 初期化
      for (let k = 0; k < d_k; k++) net.attnOut[oOff + k] = 0;
      for (let j = 0; j < T; j++) {
        const a = net.attn[sOff + i * T + j];
        const vOff = j * hd + hi * d_k;
        for (let k = 0; k < d_k; k++) {
          net.attnOut[oOff + k] += a * net.V[vOff + k];
        }
      }
    }
  }

  // (4) Y = concat(attnOut) · W_O    (concat は attnOut のメモリ配置がすでに (T, h*d_k))
  matmul(net.attnOut, T, hd, block0.W_O, d_model, net.Y);

  // (5) 残差接続: residual1 = X + Y
  for (let t = 0; t < T; t++) {
    const off = t * d_model;
    for (let d = 0; d < d_model; d++) {
      net.residual1[off + d] = net.X[off + d] + net.Y[off + d];
    }
  }

  // (6) LayerNorm 1: ln1_out = γ * (residual1 - μ) / sqrt(σ² + ε) + β
  //   per-token (= 各行ごとに d_model 軸方向で正規化)。
  //   PyTorch F.layer_norm 互換 (有偏分散、分母 N=d_model)。
  const LN_EPS = 1e-5;
  layerNormPerToken(net.residual1, T, d_model, block0.LN1_gamma, block0.LN1_beta, net.ln1_out, LN_EPS);

  // (7) FFN: ffn_pre = ln1_out · W1 + b1   (T × d_ff)
  const d_ff = net.d_ff;
  matmul(net.ln1_out, T, d_model, block0.FFN_W1, d_ff, net.ffn_pre);
  for (let t = 0; t < T; t++) {
    const off = t * d_ff;
    for (let k = 0; k < d_ff; k++) net.ffn_pre[off + k] += block0.FFN_b1[k];
  }
  // (8) ffn_h = GELU(ffn_pre)
  gelu(net.ffn_pre, net.ffn_h);
  // (9) ffn_out = ffn_h · W2 + b2   (T × d_model)
  matmul(net.ffn_h, T, d_ff, block0.FFN_W2, d_model, net.ffn_out);
  for (let t = 0; t < T; t++) {
    const off = t * d_model;
    for (let d = 0; d < d_model; d++) net.ffn_out[off + d] += block0.FFN_b2[d];
  }

  // (10) 2 回目の残差: residual2 = ln1_out + ffn_out
  for (let t = 0; t < T; t++) {
    const off = t * d_model;
    for (let d = 0; d < d_model; d++) {
      net.residual2[off + d] = net.ln1_out[off + d] + net.ffn_out[off + d];
    }
  }
  // (11) LayerNorm 2: ln2_out = LayerNorm(residual2)
  layerNormPerToken(net.residual2, T, d_model, block0.LN2_gamma, block0.LN2_beta, net.ln2_out, LN_EPS);

  net.phase = "forward";
  return net.ln2_out;
}

// per-token LayerNorm: 各行ごとに d_model 軸方向で平均/分散を取って標準化、γ/β を適用。
function layerNormPerToken(input, T, d_model, gamma, beta, output, eps) {
  for (let t = 0; t < T; t++) {
    const off = t * d_model;
    let mu = 0;
    for (let d = 0; d < d_model; d++) mu += input[off + d];
    mu /= d_model;
    let var_ = 0;
    for (let d = 0; d < d_model; d++) {
      const dx = input[off + d] - mu;
      var_ += dx * dx;
    }
    var_ /= d_model;
    const invStd = 1 / Math.sqrt(var_ + eps);
    for (let d = 0; d < d_model; d++) {
      const x_norm = (input[off + d] - mu) * invStd;
      output[off + d] = gamma[d] * x_norm + beta[d];
    }
  }
}

/**
 * 状態を「forward 前」に戻す。重みは触らない。
 */
export function reset(net) {
  net.tokens.fill(0);
  net.X.fill(0);
  net.Q.fill(0);
  net.K.fill(0);
  net.V.fill(0);
  net.scores.fill(0);
  net.attn.fill(0);
  net.attnOut.fill(0);
  net.Y.fill(0);
  net.residual1.fill(0);
  net.ln1_out.fill(0);
  net.ffn_pre.fill(0);
  net.ffn_h.fill(0);
  net.ffn_out.fill(0);
  net.residual2.fill(0);
  net.ln2_out.fill(0);
  net.phase = "idle";
}

// ─── Multi-Head アクセサ ──────────────────────────────────
// view.js / explain.js から「特定 head の (T, d_k) 部分」を取り出すヘルパ。
// メモリは net.Q etc. を共有する (subarray が view を返す)。

/**
 * net.Q / K / V の特定 head 分を (T, d_k) view として返す。
 *
 * メモリ配置: net.Q[t, hi, k] = net.Q[t * h * d_k + hi * d_k + k]
 * このとき特定 head の slice は連続ではなく、各 row が dt = h * d_k
 * 単位で step する。なので普通の subarray では取れず、コピーする。
 *
 * @param {Float64Array} buf  (T * h * d_k)
 * @param {number} T
 * @param {number} h
 * @param {number} d_k
 * @param {number} hi
 * @returns {Float64Array}  (T, d_k) を 1D 化したもの
 */
export function headSliceTHD(buf, T, h, d_k, hi) {
  const out = new Float64Array(T * d_k);
  const hd = h * d_k;
  for (let t = 0; t < T; t++) {
    const src = t * hd + hi * d_k;
    const dst = t * d_k;
    for (let k = 0; k < d_k; k++) out[dst + k] = buf[src + k];
  }
  return out;
}

/**
 * net.scores / attn の特定 head 分を (T, T) view として返す。
 * メモリ配置: net.scores[hi, i, j] = net.scores[hi * T * T + i * T + j]
 * 特定 head は連続なので subarray で view を取れる。
 *
 * @param {Float64Array} buf  (h * T * T)
 * @param {number} T
 * @param {number} hi
 * @returns {Float64Array} 長さ T*T の view
 */
export function headSliceHTT(buf, T, hi) {
  return buf.subarray(hi * T * T, (hi + 1) * T * T);
}
