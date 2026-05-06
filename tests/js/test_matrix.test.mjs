// matrix.js のユニットテスト。
// PyTorch (tests/py/reference.py) で生成した fixture と数値比較する。
// `make fixtures` で fixture を再生成、`make test-js` で実行。

import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

import {
  matmul,
  softmax,
  layerNorm,
  gelu,
  transpose,
  fromNestedArray,
  toNestedArray,
  get2,
  set2,
  row,
  addInPlace,
  scaleInPlace,
} from "../../src/js/matrix.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURE_PATH = resolve(HERE, "../fixtures/p1_matrix.json");

// 1e-12 オーダで一致を要求 (float64 の演算順序差はこれ以下に収まるはず)。
const TOL = 1e-12;

// ─── fixture 読み込み (まだ生成されていなければ skip) ─────────────

let FIX = null;
if (existsSync(FIXTURE_PATH)) {
  FIX = JSON.parse(readFileSync(FIXTURE_PATH, "utf-8"));
}

function ensureFixturesAvailable(t) {
  if (FIX === null) {
    t.skip(`Fixture not generated yet. Run: make fixtures`);
    return false;
  }
  return true;
}

function approxEqual2D(refNested, jsFlat, rows, cols, label = "") {
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const ref = refNested[r][c];
      const js = jsFlat[r * cols + c];
      if (Math.abs(js - ref) > TOL) {
        assert.fail(
          `mismatch at [${r},${c}] (${label}): js=${js}, ref=${ref}, ` +
          `diff=${Math.abs(js - ref)}`,
        );
      }
    }
  }
}

function approxEqualArr(jsArr, refArr, label = "") {
  assert.equal(jsArr.length, refArr.length, `length mismatch (${label})`);
  for (let i = 0; i < jsArr.length; i++) {
    if (Math.abs(jsArr[i] - refArr[i]) > TOL) {
      assert.fail(
        `mismatch at [${i}] (${label}): js=${jsArr[i]}, ref=${refArr[i]}`,
      );
    }
  }
}

// ─── matmul ──────────────────────────────────────────────────────

test("matmul: 全 fixture が PyTorch と 1e-12 オーダで一致", (t) => {
  if (!ensureFixturesAvailable(t)) return;
  const matFix = FIX.filter((f) => f.kind === "matmul");
  assert.ok(matFix.length > 0, "matmul fixture が無い");
  for (const fx of matFix) {
    const A = fromNestedArray(fx.A);
    const B = fromNestedArray(fx.B);
    const C = matmul(A, fx.M, fx.K, B, fx.N);
    approxEqual2D(fx.C, C, fx.M, fx.N, fx.name);
  }
});

test("matmul: out 引数で破壊的書き込み (新規確保なし)", () => {
  const A = Float64Array.from([1, 2, 3, 4]);     // 2x2
  const B = Float64Array.from([5, 6, 7, 8]);     // 2x2
  const C = new Float64Array(4);
  const ret = matmul(A, 2, 2, B, 2, C);
  assert.equal(ret, C, "out が返り値として返るべき");
  // 1*5+2*7=19, 1*6+2*8=22, 3*5+4*7=43, 3*6+4*8=50
  assert.deepStrictEqual(Array.from(C), [19, 22, 43, 50]);
});

// ─── softmax ─────────────────────────────────────────────────────

test("softmax: 全 fixture が PyTorch と 1e-12 オーダで一致", (t) => {
  if (!ensureFixturesAvailable(t)) return;
  const sm = FIX.filter((f) => f.kind === "softmax");
  assert.ok(sm.length > 0, "softmax fixture が無い");
  for (const fx of sm) {
    const x = fromNestedArray(fx.x);
    const y = softmax(x, fx.rows, fx.cols);
    approxEqual2D(fx.y, y, fx.rows, fx.cols, fx.name);
  }
});

test("softmax: 各行の和が 1.0", () => {
  const x = Float64Array.from([1, 2, 3,  -1, 0, 1]);  // 2x3
  const y = softmax(x, 2, 3);
  for (let r = 0; r < 2; r++) {
    let s = 0;
    for (let c = 0; c < 3; c++) s += y[r * 3 + c];
    assert.ok(Math.abs(s - 1.0) < 1e-15, `row ${r}: sum=${s}`);
  }
});

test("softmax: 大きい値でも数値安定 (overflow しない)", () => {
  const x = Float64Array.from([1000, 1001, 999]);   // 1x3、極端な値
  const y = softmax(x, 1, 3);
  let s = 0;
  for (let i = 0; i < 3; i++) s += y[i];
  assert.ok(Math.abs(s - 1.0) < 1e-15, `sum=${s}`);
  // 全要素が finite
  for (let i = 0; i < 3; i++) assert.ok(Number.isFinite(y[i]));
});

// ─── LayerNorm ───────────────────────────────────────────────────

test("layerNorm: 全 fixture が PyTorch と 1e-12 オーダで一致", (t) => {
  if (!ensureFixturesAvailable(t)) return;
  const ln = FIX.filter((f) => f.kind === "layernorm");
  assert.ok(ln.length > 0, "layernorm fixture が無い");
  for (const fx of ln) {
    const x = fromNestedArray(fx.x);
    const gamma = new Float64Array(fx.gamma);
    const beta = new Float64Array(fx.beta);
    const y = layerNorm(x, fx.rows, fx.cols, gamma, beta, fx.eps);
    approxEqual2D(fx.y, y, fx.rows, fx.cols, fx.name);
  }
});

test("layerNorm: gamma=1, beta=0 で各行は mean≈0, var≈1", () => {
  // 任意の入力に対して、γ=1 / β=0 / eps=1e-5 なら出力の行ごと統計が揃う。
  const cols = 8;
  const rows = 3;
  const x = new Float64Array(rows * cols);
  for (let i = 0; i < x.length; i++) x[i] = Math.sin(i * 0.7) * 5;  // 任意のばらけた値
  const gamma = new Float64Array(cols).fill(1);
  const beta = new Float64Array(cols).fill(0);
  const y = layerNorm(x, rows, cols, gamma, beta, 1e-5);
  for (let r = 0; r < rows; r++) {
    let mean = 0;
    for (let c = 0; c < cols; c++) mean += y[r * cols + c];
    mean /= cols;
    assert.ok(Math.abs(mean) < 1e-10, `row ${r}: mean=${mean} (expected ≈0)`);

    let varSum = 0;
    for (let c = 0; c < cols; c++) {
      const d = y[r * cols + c] - mean;
      varSum += d * d;
    }
    const variance = varSum / cols;
    // var は 1 にほぼなるが、eps の影響で 1 より少し小さい (= var/(var+eps))。
    // ここでは「ざっくり 1 に近い」だけ確認 (0.99..1.01 くらい)。
    assert.ok(
      Math.abs(variance - 1.0) < 0.01,
      `row ${r}: var=${variance} (expected ≈1)`,
    );
  }
});

// ─── GELU ────────────────────────────────────────────────────────

test("gelu: 全 fixture が PyTorch (tanh approximation) と 1e-12 オーダで一致", (t) => {
  if (!ensureFixturesAvailable(t)) return;
  const ge = FIX.filter((f) => f.kind === "gelu");
  assert.ok(ge.length > 0, "gelu fixture が無い");
  for (const fx of ge) {
    const x = new Float64Array(fx.x);
    const y = gelu(x);
    approxEqualArr(Array.from(y), fx.y, fx.name);
  }
});

test("gelu: gelu(0) = 0", () => {
  const x = Float64Array.from([0]);
  const y = gelu(x);
  assert.ok(Math.abs(y[0]) < 1e-15);
});

// ─── transpose ───────────────────────────────────────────────────

test("transpose: 3x4 行列の T = 4x3 行列", () => {
  const A = Float64Array.from([
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
  ]);
  const At = transpose(A, 3, 4);
  // shape (4, 3) flat layout:
  //   col 0 of A → row 0 of At: [1, 5, 9]
  //   col 1 → row 1: [2, 6, 10]
  //   col 2 → row 2: [3, 7, 11]
  //   col 3 → row 3: [4, 8, 12]
  assert.deepStrictEqual(
    Array.from(At),
    [1, 5, 9,  2, 6, 10,  3, 7, 11,  4, 8, 12],
  );
});

test("transpose: 二重転置で元に戻る", () => {
  const M = 4, N = 7;
  const A = new Float64Array(M * N);
  for (let i = 0; i < A.length; i++) A[i] = (i * 13) % 100 - 50;
  const At = transpose(A, M, N);
  const Att = transpose(At, N, M);
  assert.deepStrictEqual(Array.from(Att), Array.from(A));
});

// ─── ヘルパ ──────────────────────────────────────────────────────

test("get2 / set2: 行列セルの読み書き", () => {
  const A = new Float64Array(6);   // 2x3
  set2(A, 3, 1, 2, 7.5);
  assert.equal(get2(A, 3, 1, 2), 7.5);
  assert.equal(A[1 * 3 + 2], 7.5);
});

test("row: 行 view (subarray) は元配列を共有する", () => {
  const A = Float64Array.from([1, 2, 3, 4, 5, 6]);  // 2x3
  const r = row(A, 3, 1);
  assert.deepStrictEqual(Array.from(r), [4, 5, 6]);
  // subarray なので書き込みが元に反映される
  r[0] = 99;
  assert.equal(A[3], 99);
});

test("addInPlace: A += B", () => {
  const A = Float64Array.from([1, 2, 3]);
  const B = Float64Array.from([10, 20, 30]);
  addInPlace(A, B);
  assert.deepStrictEqual(Array.from(A), [11, 22, 33]);
});

test("scaleInPlace: A *= s", () => {
  const A = Float64Array.from([1, 2, 3]);
  scaleInPlace(A, 2.5);
  assert.deepStrictEqual(Array.from(A), [2.5, 5, 7.5]);
});

test("fromNestedArray ⇄ toNestedArray は完全な round-trip", () => {
  const orig = [[1, 2, 3], [4, 5, 6]];
  const flat = fromNestedArray(orig);
  assert.deepStrictEqual(Array.from(flat), [1, 2, 3, 4, 5, 6]);
  const back = toNestedArray(flat, 2, 3);
  assert.deepStrictEqual(back, orig);
});
