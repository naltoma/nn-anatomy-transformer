// rng.js — 単一マスター seed から独立なサブストリームを導出する。
// 設計書 §6.5 / §6.6 に対応。ES modules。

/**
 * サブストリーム tag。将来 v2 でバッチシャッフル等を足すときは
 * ここに SHUFFLE などを追記するだけ。
 */
export const SUB = Object.freeze({
  WEIGHTS: 0xa5a5a5a5,
  DATA: 0x5a5a5a5a,
});

/**
 * Mulberry32: 32-bit 状態の軽量 PRNG。
 * seed → [0,1) を返す関数 (以降 rng と呼ぶ)。
 * 同じ seed からは必ず同じ列を返す (bitwise 一致)。
 */
export function mulberry32(seed) {
  let t = seed >>> 0;
  return function () {
    t = (t + 0x6d2b79f5) >>> 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * 標準正規分布からのサンプリング (Box-Muller)。
 * rng は mulberry32() などの [0,1) 返す関数。
 * 1 回呼ぶごとに rng を 2 回消費する。
 */
export function randn(rng) {
  const u1 = Math.max(rng(), 1e-12); // log(0) ガード
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/**
 * マスター seed と tag からサブシードを導出する。
 * XOR だけ。Mulberry32 側の状態発散が十分カオティックなので相関は実用上ゼロ。
 */
export function deriveSubSeed(masterSeed, tag) {
  return ((masterSeed >>> 0) ^ (tag >>> 0)) >>> 0;
}

/**
 * ユーザーがシードを指定しなかったときの自動 seed。
 * Date.now() は秒単位で粗いので Math.random() と XOR して混ぜる。
 * 本関数の戻り値は必ず seedUsed として記録し、ログ・JSON に残すこと。
 */
export function autoSeed() {
  return (Date.now() ^ Math.floor(Math.random() * 2 ** 31)) >>> 0;
}
