// Tier 1 model.js (createTransformer + forward) のユニットテスト。
// presets/japanese-mini-v1.json から inline された DEFAULT_PRESET を使い、
// 8 例文すべてで forward を実行して numpy 参照 (tests/fixtures/p3_attention.json)
// と数値一致 (1e-12 オーダ) するかを検証する。

import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

import { createTransformer, forward, reset } from "../../src/js/model.js";
import { DEFAULT_PRESET } from "../../src/js/presets.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURE_PATH = resolve(HERE, "../fixtures/p3_attention.json");

const TOL = 1e-12;

let FIX = null;
if (existsSync(FIXTURE_PATH)) {
  FIX = JSON.parse(readFileSync(FIXTURE_PATH, "utf-8"));
}

function ensureFixturesAvailable(t) {
  if (FIX === null) {
    t.skip("Fixture not generated yet. Run: make fixtures");
    return false;
  }
  return true;
}

function approxEqualFlat(refFlat, jsFlat, label = "") {
  if (refFlat.length !== jsFlat.length) {
    assert.fail(
      `length mismatch (${label}): ref=${refFlat.length}, js=${jsFlat.length}`,
    );
  }
  for (let i = 0; i < refFlat.length; i++) {
    const ref = refFlat[i];
    const js = jsFlat[i];
    if (Math.abs(js - ref) > TOL) {
      assert.fail(
        `mismatch at [${i}] (${label}): js=${js}, ref=${ref}, diff=${Math.abs(js - ref)}`,
      );
    }
  }
}

// ─── 構造的なテスト (fixture 不要) ────────────────────────────

test("createTransformer: DEFAULT_PRESET から net を構築できる (Tier 2: h=2, d_k=8)", () => {
  const net = createTransformer({ preset: DEFAULT_PRESET });
  assert.equal(net.T, 5);
  assert.equal(net.d_model, 16);
  assert.equal(net.d_k, 8);     // Tier 2: d_model / h = 16 / 2 = 8
  assert.equal(net.h, 2);
  assert.equal(net.N, 1);
  assert.equal(net.vocabSize, 16);
  assert.equal(net.vocab.length, 16);
  // 重みのサイズ
  assert.equal(net.W_E.length, 16 * 16);            // vocab × d_model
  assert.equal(net.W_P.length, 5 * 16);             // T × d_model
  assert.equal(net.blocks.length, 1);
  assert.equal(net.blocks[0].W_Q.length, 16 * 16);  // d_model × (h*d_k) = 16 × 16
  assert.equal(net.blocks[0].W_K.length, 16 * 16);
  assert.equal(net.blocks[0].W_V.length, 16 * 16);
  assert.equal(net.blocks[0].W_O.length, 16 * 16);  // (h*d_k) × d_model
  // 中間結果バッファ
  assert.equal(net.X.length, 5 * 16);
  assert.equal(net.Q.length, 5 * 2 * 8);            // T × h × d_k
  assert.equal(net.K.length, 5 * 2 * 8);
  assert.equal(net.V.length, 5 * 2 * 8);
  assert.equal(net.scores.length, 2 * 5 * 5);       // h × T × T
  assert.equal(net.attn.length, 2 * 5 * 5);
  assert.equal(net.attnOut.length, 5 * 2 * 8);
  assert.equal(net.Y.length, 5 * 16);
  assert.equal(net.residual1.length, 5 * 16);       // 残差接続出力
  assert.equal(net.ln1_out.length, 5 * 16);         // LayerNorm 出力
  // LayerNorm パラメータ (γ=1, β=0)
  assert.equal(net.blocks[0].LN1_gamma.length, 16);
  assert.equal(net.blocks[0].LN1_beta.length, 16);
  for (const v of net.blocks[0].LN1_gamma) assert.equal(v, 1.0);
  for (const v of net.blocks[0].LN1_beta) assert.equal(v, 0.0);
  // FFN 構造
  assert.equal(net.d_ff, 32);
  assert.equal(net.blocks[0].FFN_W1.length, 16 * 32);  // (d_model, d_ff)
  assert.equal(net.blocks[0].FFN_b1.length, 32);
  assert.equal(net.blocks[0].FFN_W2.length, 32 * 16);  // (d_ff, d_model)
  assert.equal(net.blocks[0].FFN_b2.length, 16);
  assert.equal(net.ffn_pre.length, 5 * 32);
  assert.equal(net.ffn_h.length, 5 * 32);
  assert.equal(net.ffn_out.length, 5 * 16);
  assert.equal(net.residual2.length, 5 * 16);
  assert.equal(net.ln2_out.length, 5 * 16);
  // 初期状態
  assert.equal(net.phase, "idle");
});

test("createTransformer: presetMeta に hand-crafted の設計意図が入っている (Tier 2)", () => {
  const net = createTransformer({ preset: DEFAULT_PRESET });
  assert.equal(net.presetName, "japanese-mini-v1");
  assert.equal(net.presetMeta.kind, "hand-crafted");
  assert.equal(net.presetMeta.tier, 2);
  // designIntent.attentionBonds に bond の意図が記載
  const bonds = net.presetMeta.designIntent.attentionBonds;
  assert.ok(bonds["0"].includes("adjective"));
  assert.ok(bonds["1"].includes("pronoun"));
  assert.ok(bonds["2"].includes("predicate"));
});

test("createTransformer: h*d_k != d_model の preset は拒否", () => {
  const bad = JSON.parse(JSON.stringify(DEFAULT_PRESET));
  bad.config.h = 3;     // 3 * 8 != 16
  assert.throws(() => createTransformer({ preset: bad }), /h\*d_k/);
});

test("forward: tokens の長さが T と違うと例外", () => {
  const net = createTransformer({ preset: DEFAULT_PRESET });
  assert.throws(() => forward(net, [0, 1, 2]), /tokens\.length=3/);
});

test("forward: 範囲外 token id は例外", () => {
  const net = createTransformer({ preset: DEFAULT_PRESET });
  assert.throws(() => forward(net, [0, 1, 99, 3, 4]), /は範囲外/);
});

test("forward: 各 head の各行の attn の和が 1.0 (softmax の不変条件)", () => {
  const net = createTransformer({ preset: DEFAULT_PRESET });
  forward(net, [0, 12, 8, 4, 15]); // これ は 美しい 花 です
  const T = net.T, H = net.h;
  for (let hi = 0; hi < H; hi++) {
    for (let i = 0; i < T; i++) {
      let s = 0;
      for (let j = 0; j < T; j++) s += net.attn[hi * T * T + i * T + j];
      assert.ok(Math.abs(s - 1.0) < 1e-12, `head ${hi} row ${i}: sum=${s}`);
    }
  }
});

test("forward: phase が 'forward' に変わる", () => {
  const net = createTransformer({ preset: DEFAULT_PRESET });
  assert.equal(net.phase, "idle");
  forward(net, [0, 12, 8, 4, 15]);
  assert.equal(net.phase, "forward");
});

test("reset: forward 後の中間結果がクリアされる", () => {
  const net = createTransformer({ preset: DEFAULT_PRESET });
  forward(net, [0, 12, 8, 4, 15]);
  // forward 後は attn に softmax 結果が入っている
  assert.ok(net.attn.some((v) => v > 0));
  reset(net);
  // クリア後はすべて 0、phase は idle
  assert.equal(net.phase, "idle");
  for (const v of net.attn) assert.equal(v, 0);
  for (const v of net.X) assert.equal(v, 0);
  for (const v of net.residual1) assert.equal(v, 0);
  for (const v of net.ln1_out) assert.equal(v, 0);
});

// ─── 数値一致テスト (numpy fixture との比較) ──────────────────

test("forward: 8 例文すべての中間結果が numpy 参照と 1e-12 で一致", (t) => {
  if (!ensureFixturesAvailable(t)) return;
  const net = createTransformer({ preset: DEFAULT_PRESET });
  // fixture は ravel された 1D 配列。JS の Float64Array と直接比較する。
  for (const fx of FIX) {
    forward(net, fx.tokens);
    approxEqualFlat(fx.X, net.X, `${fx.name}: X`);
    approxEqualFlat(fx.Q, net.Q, `${fx.name}: Q`);
    approxEqualFlat(fx.K, net.K, `${fx.name}: K`);
    approxEqualFlat(fx.V, net.V, `${fx.name}: V`);
    approxEqualFlat(fx.scores, net.scores, `${fx.name}: scores`);
    approxEqualFlat(fx.attn, net.attn, `${fx.name}: attn`);
    approxEqualFlat(fx.attnOut, net.attnOut, `${fx.name}: attnOut`);
    approxEqualFlat(fx.Y, net.Y, `${fx.name}: Y`);
    approxEqualFlat(fx.residual1, net.residual1, `${fx.name}: residual1`);
    approxEqualFlat(fx.ln1_out, net.ln1_out, `${fx.name}: ln1_out`);
    approxEqualFlat(fx.ffn_pre, net.ffn_pre, `${fx.name}: ffn_pre`);
    approxEqualFlat(fx.ffn_h, net.ffn_h, `${fx.name}: ffn_h`);
    approxEqualFlat(fx.ffn_out, net.ffn_out, `${fx.name}: ffn_out`);
    approxEqualFlat(fx.residual2, net.residual2, `${fx.name}: residual2`);
    approxEqualFlat(fx.ln2_out, net.ln2_out, `${fx.name}: ln2_out`);
  }
});

// ─── 設計意図の確認 (attention pattern が 0.5 以上で出る) ────

// Multi-Head 対応: net.attn shape は (h, T, T)、index は hi*T*T + i*T + j。
function maxAttnAcrossHeads(net, i, j) {
  const T = net.T, H = net.h;
  let best = -Infinity;
  for (let hi = 0; hi < H; hi++) {
    const v = net.attn[hi * T * T + i * T + j];
    if (v > best) best = v;
  }
  return best;
}

test("forward: 「美しい→花」の attention (head 0 = adj→noun) が 0.5 以上 (例文 1)", (t) => {
  if (!ensureFixturesAvailable(t)) return;
  const net = createTransformer({ preset: DEFAULT_PRESET });
  forward(net, [0, 12, 8, 4, 15]); // これ は 美しい 花 です
  const v = maxAttnAcrossHeads(net, 2, 3); // 美しい(t=2) → 花(s=3)
  assert.ok(v >= 0.5, `max-head attn[美しい→花] = ${v.toFixed(3)} < 0.5`);
});

test("forward: 「私→猫」の attention (head 0 = pron→noun) が 0.5 以上 (例文 3)", (t) => {
  if (!ensureFixturesAvailable(t)) return;
  const net = createTransformer({ preset: DEFAULT_PRESET });
  forward(net, [2, 12, 3, 13, 11]); // 私 は 猫 が 好き
  const v = maxAttnAcrossHeads(net, 0, 2);
  assert.ok(v >= 0.5, `max-head attn[私→猫] = ${v.toFixed(3)} < 0.5`);
});

test("forward: 「読む→本」の attention (head 1 = pred→noun) が 0.5 以上 (例文 4)", (t) => {
  if (!ensureFixturesAvailable(t)) return;
  const net = createTransformer({ preset: DEFAULT_PRESET });
  forward(net, [2, 12, 5, 14, 10]); // 私 は 本 を 読む
  const v = maxAttnAcrossHeads(net, 4, 2); // 読む(t=4) → 本(s=2)
  assert.ok(v >= 0.5, `max-head attn[読む→本] = ${v.toFixed(3)} < 0.5`);
});
