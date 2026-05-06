// matrix.js — flat-array ベースの行列演算ヘルパ。
//
// すべての行列は Float64Array の 1 次元配列で、(rows, cols) の組と一緒に渡す。
// アクセスは A[r * cols + c]。
// このスタイルは Transformer のような matmul 中心の計算で速く、
// 後で WebGPU / SIMD 化する場合にも転用しやすい。
//
// 数値検証は tests/js/test_matrix.test.mjs で PyTorch fixture (float64) と
// 1e-12 オーダで一致することを担保している。

/**
 * 行列積 C = A · B
 *   A: shape (M, K), B: shape (K, N) → C: shape (M, N)
 * すべて Float64Array。out が省略されたら新しく確保して返す。
 *
 * 実装は素朴な 3 重ループ。教育用の規模 (T=5, d_model=16) では十分。
 */
export function matmul(A, M, K, B, N, out) {
  const C = out ?? new Float64Array(M * N);
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      let s = 0;
      for (let k = 0; k < K; k++) {
        s += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = s;
    }
  }
  return C;
}

/**
 * 行ごとの softmax。x: (rows, cols) → 各行を独立に softmax。
 * 数値安定化のため、行ごとの max を引いてから exp する (overflow 防止)。
 * PyTorch の torch.softmax(x, dim=-1) と数値一致する。
 */
export function softmax(x, rows, cols, out) {
  const y = out ?? new Float64Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    const off = r * cols;
    // 行ごとの max
    let m = -Infinity;
    for (let c = 0; c < cols; c++) {
      const v = x[off + c];
      if (v > m) m = v;
    }
    // exp(x - m) と総和
    let s = 0;
    for (let c = 0; c < cols; c++) {
      const e = Math.exp(x[off + c] - m);
      y[off + c] = e;
      s += e;
    }
    // 正規化
    const inv = 1 / s;
    for (let c = 0; c < cols; c++) {
      y[off + c] *= inv;
    }
  }
  return y;
}

/**
 * 行ごとの LayerNorm (PyTorch の F.layer_norm と同等):
 *   mean   = sum(x_r) / cols
 *   var    = sum((x_r - mean)^2) / cols    (バイアス補正なし、PyTorch と同じ)
 *   y_r    = (x_r - mean) / sqrt(var + eps) * gamma + beta
 *
 * gamma, beta は長さ cols のベクトル (Float64Array)。
 * eps のデフォルトは PyTorch と揃えて 1e-5。
 *
 * 注: PyTorch の var は分母が cols (有偏) なので Math.sqrt(var + eps) は
 *     1/cols の分散基準。これに合わせるのが大事 (1/(cols-1) ではない)。
 */
export function layerNorm(x, rows, cols, gamma, beta, eps = 1e-5, out) {
  const y = out ?? new Float64Array(rows * cols);
  const invCols = 1 / cols;
  for (let r = 0; r < rows; r++) {
    const off = r * cols;
    // mean
    let mean = 0;
    for (let c = 0; c < cols; c++) mean += x[off + c];
    mean *= invCols;
    // variance
    let varSum = 0;
    for (let c = 0; c < cols; c++) {
      const d = x[off + c] - mean;
      varSum += d * d;
    }
    const variance = varSum * invCols;
    const invStd = 1 / Math.sqrt(variance + eps);
    // normalize + scale + shift
    for (let c = 0; c < cols; c++) {
      y[off + c] = (x[off + c] - mean) * invStd * gamma[c] + beta[c];
    }
  }
  return y;
}

/**
 * GELU 活性化 (Gaussian Error Linear Unit)、tanh 近似版。
 *
 *   gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
 *
 * PyTorch では `F.gelu(x, approximate="tanh")` がこの式と完全一致。
 * 教材として精度より読みやすさを優先し tanh 近似を採用。必要なら erf 版に
 * 切り替え可能 (未実装)。
 */
const GELU_COEF = Math.sqrt(2 / Math.PI);
export function gelu(x, out) {
  const n = x.length;
  const y = out ?? new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const v = x[i];
    const inner = GELU_COEF * (v + 0.044715 * v * v * v);
    y[i] = 0.5 * v * (1 + Math.tanh(inner));
  }
  return y;
}

// ─── ヘルパ ──────────────────────────────────────────────

/**
 * (rows, cols) の行列で (r, c) 番地を読む。
 */
export function get2(A, cols, r, c) {
  return A[r * cols + c];
}

/**
 * (rows, cols) の行列で (r, c) 番地に書く。
 */
export function set2(A, cols, r, c, v) {
  A[r * cols + c] = v;
}

/**
 * 行 r の view (Float64Array.subarray、コピーなし)。
 */
export function row(A, cols, r) {
  return A.subarray(r * cols, (r + 1) * cols);
}

/**
 * 行列の転置 At = A^T。A: (M, N) → At: (N, M)
 */
export function transpose(A, M, N, out) {
  const At = out ?? new Float64Array(N * M);
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      At[j * M + i] = A[i * N + j];
    }
  }
  return At;
}

/**
 * 要素ごとの加算 A += B (in-place)。A と B は同じ長さ。
 */
export function addInPlace(A, B, len) {
  const n = len ?? A.length;
  for (let i = 0; i < n; i++) A[i] += B[i];
}

/**
 * 要素ごとのスカラー倍 A *= s (in-place)。
 */
export function scaleInPlace(A, s, len) {
  const n = len ?? A.length;
  for (let i = 0; i < n; i++) A[i] *= s;
}

/**
 * (M, N) の入れ子配列を flat Float64Array に変換。
 * テスト fixture (JSON) からの読み込みに使う。
 */
export function fromNestedArray(arr2d) {
  const M = arr2d.length;
  const N = M > 0 ? arr2d[0].length : 0;
  const out = new Float64Array(M * N);
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      out[i * N + j] = arr2d[i][j];
    }
  }
  return out;
}

/**
 * flat Float64Array を (M, N) の入れ子配列に変換。
 * テスト出力比較や JSON シリアライズに使う。
 */
export function toNestedArray(flat, M, N) {
  const out = new Array(M);
  for (let i = 0; i < M; i++) {
    const r = new Array(N);
    for (let j = 0; j < N; j++) r[j] = flat[i * N + j];
    out[i] = r;
  }
  return out;
}
