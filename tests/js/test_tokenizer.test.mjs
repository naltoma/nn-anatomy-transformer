// tokenizer.js のユニットテスト。
// 設計書 §6.1 vocabulary 表と §6.2 サンプル文に対応。

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  VOCAB,
  VOCAB_SIZE,
  word2id,
  id2word,
  encode,
  decode,
  SAMPLE_SENTENCES,
} from "../../src/js/tokenizer.js";

test("VOCAB: 16 語ちょうどある", () => {
  assert.equal(VOCAB.length, 16);
  assert.equal(VOCAB_SIZE, 16);
});

test("VOCAB: 設計書 §6.1 の 16 語が全部入っている", () => {
  const expected = [
    "これ", "それ", "私",
    "猫", "花", "本", "小説",
    "大きい", "美しい", "人気",
    "読む", "好き",
    "は", "が", "を",
    "です",
  ];
  for (const w of expected) {
    assert.ok(VOCAB.includes(w), `${w} が VOCAB に無い`);
  }
});

test("VOCAB は frozen で変更不可", () => {
  // Object.freeze 済みなので push は失敗 (strict mode で TypeError)
  // ES module は strict mode 既定なので例外が飛ぶ
  assert.throws(() => VOCAB.push("追加"));
});

test("word2id ⇄ id2word は完全な round-trip", () => {
  for (let i = 0; i < VOCAB.length; i++) {
    assert.equal(word2id(id2word(i)), i);
    assert.equal(id2word(word2id(VOCAB[i])), VOCAB[i]);
  }
});

test("word2id: 設計書 §6.1 と一致した ID 番号", () => {
  // 順序が design.md と狂うと preset の互換性が壊れるので固定する。
  assert.equal(word2id("これ"), 0);
  assert.equal(word2id("それ"), 1);
  assert.equal(word2id("私"), 2);
  assert.equal(word2id("猫"), 3);
  assert.equal(word2id("花"), 4);
  assert.equal(word2id("本"), 5);
  assert.equal(word2id("小説"), 6);
  assert.equal(word2id("大きい"), 7);
  assert.equal(word2id("美しい"), 8);
  assert.equal(word2id("人気"), 9);
  assert.equal(word2id("読む"), 10);
  assert.equal(word2id("好き"), 11);
  assert.equal(word2id("は"), 12);
  assert.equal(word2id("が"), 13);
  assert.equal(word2id("を"), 14);
  assert.equal(word2id("です"), 15);
});

test("encode / decode はサンプル文で完全往復", () => {
  const sentence = ["これ", "は", "美しい", "花", "です"];
  const ids = encode(sentence);
  assert.ok(ids instanceof Int32Array);
  assert.equal(ids.length, 5);
  // 0, 12, 8, 4, 15
  assert.equal(ids[0], 0);
  assert.equal(ids[1], 12);
  assert.equal(ids[2], 8);
  assert.equal(ids[3], 4);
  assert.equal(ids[4], 15);
  const back = decode(ids);
  assert.deepStrictEqual(back, sentence);
});

test("encode: 語彙外の単語でエラー", () => {
  assert.throws(() => encode(["未知"]), /Unknown word/);
  assert.throws(() => encode(["これ", "未知"]), /Unknown word/);
});

test("id2word: 範囲外 ID でエラー", () => {
  assert.throws(() => id2word(-1), /out of range/);
  assert.throws(() => id2word(16), /out of range/);
  assert.throws(() => id2word(1.5), /out of range/);
  assert.throws(() => id2word(NaN), /out of range/);
});

test("SAMPLE_SENTENCES: 設計書 §6.2 の 8 文がある", () => {
  assert.equal(SAMPLE_SENTENCES.length, 8);
  for (const s of SAMPLE_SENTENCES) {
    assert.equal(s.length, 5, `${s.join(" ")} は T=5 ではない`);
    // 全単語がエンコード可能 (= VOCAB 内) であること
    const ids = encode(s);
    assert.equal(ids.length, 5);
  }
});

test("SAMPLE_SENTENCES: 設計書 §6.2 の文 1 (「これ は 美しい 花 です」)", () => {
  // ID 列で: [0, 12, 8, 4, 15]
  const ids = encode(SAMPLE_SENTENCES[0]);
  assert.deepStrictEqual(Array.from(ids), [0, 12, 8, 4, 15]);
});

test("SAMPLE_SENTENCES: 設計書 §6.2 の文 4 (「私 は 本 を 読む」)", () => {
  // ID 列で: [2, 12, 5, 14, 10]
  const ids = encode(SAMPLE_SENTENCES[3]);
  assert.deepStrictEqual(Array.from(ids), [2, 12, 5, 14, 10]);
});
