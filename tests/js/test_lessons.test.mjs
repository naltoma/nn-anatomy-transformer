// lessons.js の整合性テスト。設計書 §9 (Lesson curriculum) の Tier 1 部分 (T1〜T5) に対応。

import { test } from "node:test";
import assert from "node:assert/strict";

import { LESSONS, getLesson } from "../../src/js/lessons.js";
import { SAMPLE_SENTENCES } from "../../src/js/tokenizer.js";

const VIEW_TABS = ["embed", "qkv", "attn", "out", "ffn"];

test("LESSONS: Lesson 1〜Lesson 10 の 10 つがある", () => {
  assert.equal(LESSONS.length, 10);
  for (let i = 0; i < 10; i++) {
    assert.equal(LESSONS[i].id, `Lesson ${i + 1}`);
  }
});

test("LESSONS: 各 Lesson が必須フィールドを持ち、整合する", () => {
  for (const lsn of LESSONS) {
    assert.ok(lsn.id && /^Lesson (?:[1-9]|10)$/.test(lsn.id), `bad id: ${lsn.id}`);
    assert.ok(typeof lsn.title === "string" && lsn.title.length > 0);
    assert.ok(typeof lsn.hint === "string" && lsn.hint.length > 0);
    assert.ok(Array.isArray(lsn.checks) && lsn.checks.length === 3,
      `${lsn.id}: checks は 3 件必要`);
    for (const c of lsn.checks) {
      assert.ok(typeof c === "string" && c.length > 0);
    }
    // sampleIdx は SAMPLE_SENTENCES の範囲内
    assert.ok(Number.isInteger(lsn.sampleIdx)
      && lsn.sampleIdx >= 0
      && lsn.sampleIdx < SAMPLE_SENTENCES.length,
      `${lsn.id}: sampleIdx=${lsn.sampleIdx} が範囲外`);
    // viewTab は許容セットに含まれる
    assert.ok(VIEW_TABS.includes(lsn.viewTab),
      `${lsn.id}: viewTab=${lsn.viewTab} が不正`);
    // runForward は boolean
    assert.equal(typeof lsn.runForward, "boolean");
  }
});

test("getLesson: 既知 id で取れる、未知 id で null", () => {
  for (const lsn of LESSONS) {
    assert.equal(getLesson(lsn.id), lsn);
  }
  assert.equal(getLesson("Lesson 99"), null);
  assert.equal(getLesson(""), null);
});

test("LESSONS: 同じ id が重複していない", () => {
  const ids = new Set();
  for (const lsn of LESSONS) {
    assert.ok(!ids.has(lsn.id), `duplicate id: ${lsn.id}`);
    ids.add(lsn.id);
  }
});

test("Lesson 1: Embedding を見るレッスン (viewTab=embed, runForward=false)", () => {
  const lsn = getLesson("Lesson 1");
  assert.equal(lsn.viewTab, "embed");
  assert.equal(lsn.runForward, false); // Lesson 1 は X 観察だけで足りる
});

test("Lesson 3 / Lesson 4 / Lesson 5: Q/K/V projection (viewTab=qkv, runForward=true)", () => {
  for (const id of ["Lesson 3", "Lesson 4", "Lesson 5"]) {
    const lsn = getLesson(id);
    assert.equal(lsn.viewTab, "qkv");
    assert.equal(lsn.runForward, true);
  }
});

test("Lesson 6 / Lesson 7: Attention 系 (viewTab=attn, runForward=true)", () => {
  for (const id of ["Lesson 6", "Lesson 7"]) {
    const lsn = getLesson(id);
    assert.equal(lsn.viewTab, "attn");
    assert.equal(lsn.runForward, true);
  }
});

test("Lesson 6 / Lesson 7 の sampleIdx は「これ は 美しい 花 です」(= index 0)", () => {
  // 設計意図: Lesson 6 で scores、Lesson 7 で softmax 後の attn を同じサンプルで観察。
  for (const id of ["Lesson 6", "Lesson 7"]) {
    const lsn = getLesson(id);
    assert.equal(lsn.sampleIdx, 0);
    assert.deepStrictEqual(
      [...SAMPLE_SENTENCES[lsn.sampleIdx]],
      ["これ", "は", "美しい", "花", "です"],
    );
  }
});

test("Lesson 8: Multi-Head の sampleIdx は #3「私 は 猫 が 好き」(= index 2)", () => {
  // 設計意図: 本シミュレータで唯一、head 0 (pron→noun) と head 1 (pred→noun) が
  // 同時に強く立つサンプル。Multi-Head の役割分担を観察するのに必須。
  const lsn = getLesson("Lesson 8");
  assert.equal(lsn.sampleIdx, 2);
  assert.deepStrictEqual(
    [...SAMPLE_SENTENCES[lsn.sampleIdx]],
    ["私", "は", "猫", "が", "好き"],
  );
  assert.equal(lsn.viewTab, "attn");
  assert.equal(lsn.runForward, true);
});

test("Lesson 9: 残差 + LN は Out タブで sample #3 の T0「私」を題材に観察", () => {
  // Lesson 8 で head 0 attn[T0,T2]=0.977 を観察したので、その続きで残差を見る。
  const lsn = getLesson("Lesson 9");
  assert.equal(lsn.sampleIdx, 2);
  assert.equal(lsn.viewTab, "out");
  assert.equal(lsn.runForward, true);
});

test("Lesson 10: FFN は FFN タブで sample #3 を題材に検出器ニューロンを観察", () => {
  const lsn = getLesson("Lesson 10");
  assert.equal(lsn.sampleIdx, 2);
  assert.equal(lsn.viewTab, "ffn");
  assert.equal(lsn.runForward, true);
});
