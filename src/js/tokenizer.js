// tokenizer.js — 16 語固定の日本語語彙。
// 設計書 §6.1 vocabulary 表に対応。長さ T=5 のサンプル文 (§6.2) も同梱。

/**
 * 語彙 (id 0..15)。順序は設計書 §6.1 と同じ。
 *   0-2: 指示詞・代名詞
 *   3-6: 名詞
 *   7-9: 形容詞
 *   10-11: 動詞
 *   12-14: 助詞
 *   15: コピュラ
 */
export const VOCAB = Object.freeze([
  "これ",   //  0
  "それ",   //  1
  "私",     //  2
  "猫",     //  3
  "花",     //  4
  "本",     //  5
  "小説",   //  6
  "大きい", //  7
  "美しい", //  8
  "人気",   //  9
  "読む",   // 10
  "好き",   // 11
  "は",     // 12
  "が",     // 13
  "を",     // 14
  "です",   // 15
]);

export const VOCAB_SIZE = VOCAB.length;

/** 単語 → ID の逆引き表。Object.create(null) で prototype 汚染を避ける。 */
const WORD2ID = (() => {
  const m = Object.create(null);
  for (let i = 0; i < VOCAB.length; i++) m[VOCAB[i]] = i;
  return Object.freeze(m);
})();

/**
 * 単語を ID に。語彙外の単語が来たらエラー。
 * @param {string} word
 * @returns {number}
 */
export function word2id(word) {
  if (!(word in WORD2ID)) {
    throw new Error(`Unknown word: ${JSON.stringify(word)}`);
  }
  return WORD2ID[word];
}

/**
 * ID を単語に。範囲外の ID が来たらエラー。
 * @param {number} id
 * @returns {string}
 */
export function id2word(id) {
  if (!Number.isInteger(id) || id < 0 || id >= VOCAB.length) {
    throw new Error(`id out of range: ${id}`);
  }
  return VOCAB[id];
}

/**
 * 単語列を ID 列 (Int32Array) に変換。
 * @param {string[]} words
 * @returns {Int32Array}
 */
export function encode(words) {
  const out = new Int32Array(words.length);
  for (let i = 0; i < words.length; i++) out[i] = word2id(words[i]);
  return out;
}

/**
 * ID 列を単語列に変換。
 * @param {Int32Array | number[]} ids
 * @returns {string[]}
 */
export function decode(ids) {
  const out = [];
  for (let i = 0; i < ids.length; i++) out.push(id2word(ids[i]));
  return out;
}

/**
 * 設計書 §6.2 のサンプル文 8 つ (T=5 固定)。
 * デフォルト preset (japanese-mini-v1) もこれら 8 文の next-token prediction で
 * 学習される。UI の Sentence セレクタからこれらを直接選べるようにする想定。
 */
export const SAMPLE_SENTENCES = Object.freeze([
  Object.freeze(["これ", "は", "美しい", "花", "です"]),
  Object.freeze(["それ", "は", "人気", "小説", "です"]),
  Object.freeze(["私",   "は", "猫",     "が", "好き"]),
  Object.freeze(["私",   "は", "本",     "を", "読む"]),
  Object.freeze(["私",   "は", "小説",   "を", "読む"]),
  Object.freeze(["大きい", "猫", "は", "美しい", "です"]),
  Object.freeze(["美しい", "花", "は", "人気",   "です"]),
  Object.freeze(["これ", "は", "大きい", "本", "です"]),
]);
