// explain.js — 「式の展開」HTML テンプレート。
// 設計書 §7.3 に対応。MLP 版 v1 の哲学を継承し、
// [段1] 一般形 → [段2] 当てはめ → [段3] 数値 の 3 段板書を作る。
//
// 引数: state = { net, selection: { matrix, row, col } }
//   matrix は文字列キー: preset 由来の重み行列 ("W_E" | "W_P" | "W_Q" | "W_K" | "W_V")
//                        または forward 由来の中間値 ("X" | "Q" | "K" | "V" | "scores" | "attn" | "attnOut" | "Y")
//
// 戻り値: HTML 文字列 (controller.js が ui.explainBody.innerHTML に流し込む)。

// ─── 数式レンダリング用ヘルパ ─────────────────────────────

export function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

/**
 * Lesson 文中で使う、ごく軽量な markdown 風変換 + HTML エスケープ。
 *   1. まず HTML エスケープ
 *   2. その上で **xxx** を <b>xxx</b> に置換 (太字強調)
 * 改行は CSS の white-space: pre-line で処理する想定なので、ここでは触らない。
 */
export function escapeAndBold(s) {
  return escapeHtml(s).replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
}

// 変数 (italic) + subscript + superscript
function mv(name, sub, sup) {
  const n = `<i>${escapeHtml(name)}</i>`;
  const sb = (sub != null && sub !== "") ? `<sub>${escapeHtml(sub)}</sub>` : "";
  const sp = (sup != null && sup !== "") ? `<sup>${escapeHtml(sup)}</sup>` : "";
  return n + sb + sp;
}
function mn(v)  { return `<span class="mnum">${escapeHtml(v)}</span>`; }
function mo(op) { return `<span class="mop">${escapeHtml(op)}</span>`; }
function mline(html)        { return `<div class="line">${html}</div>`; }
function mblock(...lines)   { return `<div class="math">${lines.join("")}</div>`; }

function fmt(v, digits = 3) {
  if (!Number.isFinite(v)) return String(v);
  return (v >= 0 ? "+" : "") + v.toFixed(digits);
}
function fmtAttn(v, digits = 3) {
  if (!Number.isFinite(v)) return String(v);
  return v.toFixed(digits);
}

// 「T2 美しい」のようなラベル。token id→単語。
function tokenLabel(net, t) {
  const word = net.vocab[net.tokens[t]] ?? "?";
  return `T${t} ${word}`;
}

// ─── エントリポイント ─────────────────────────────────────

export function renderExplainHtml(state) {
  const sel = state?.selection;
  if (!sel) {
    return `
      <p class="muted">
        中央パネルの行列のセルをクリックすると、ここにそのセルの計算式が
        [段1] 一般形 → [段2] 当てはめ → [段3] 数値 の 3 段で展開されます。
      </p>`;
  }
  const { net } = state;
  if (!net) return "";

  const phase = net.phase ?? "idle";
  // 行列キー正規化: "Q_h0" → { name: "Q", head: 0 }、"Q" → { name: "Q", head: null }
  const { name, head } = parseMatrixKey(sel.matrix);

  // forward 前でも見られる行列: 入力埋め込み X (precomputed) と preset 由来の重み行列。
  const PRESET_OK_AT_IDLE = new Set([
    "X", "W_E", "W_P", "W_Q", "W_K", "W_V", "FFN_W1", "FFN_W2",
  ]);
  if (phase === "idle" && !PRESET_OK_AT_IDLE.has(name)) {
    return `
      <p class="muted">
        ${escapeHtml(sel.matrix)}[${sel.row},${sel.col}] はまだ Forward が実行されていません。
        上部の <b>Forward</b> ボタンを押してから再度クリックしてください。
      </p>`;
  }

  switch (name) {
    case "W_E":     return renderWE(net, sel);
    case "W_P":     return renderWP(net, sel);
    case "W_Q":     return renderProjectionWeight(net, sel, "Q");
    case "W_K":     return renderProjectionWeight(net, sel, "K");
    case "W_V":     return renderProjectionWeight(net, sel, "V");
    case "X":       return renderEmbed(net, sel);
    case "Q":       return renderQKV(net, sel, "Q", head);
    case "K":       return renderQKV(net, sel, "K", head);
    case "V":       return renderQKV(net, sel, "V", head);
    case "scores":  return renderScores(net, sel, head);
    case "attn":    return renderAttn(net, sel, head);
    case "attnOut": return renderAttnOut(net, sel, head);
    case "Y":       return renderY(net, sel);
    case "residual1": return renderResidual1(net, sel);
    case "ln1_out":   return renderLN1(net, sel);
    case "FFN_W1":    return renderFFNWeight(net, sel, "W1");
    case "FFN_W2":    return renderFFNWeight(net, sel, "W2");
    case "ffn_h":     return renderFFNHidden(net, sel);
    case "ffn_out":   return renderFFNOut(net, sel);
    case "residual2": return renderResidual2(net, sel);
    case "ln2_out":   return renderLN2(net, sel);
    default:
      return `<p class="muted">未対応の行列: ${escapeHtml(sel.matrix)}</p>`;
  }
}

/** "Q_h0" → { name: "Q", head: 0 }、"Q" → { name: "Q", head: null } */
function parseMatrixKey(key) {
  const m = /^([A-Za-z_]+)_h(\d+)$/.exec(key);
  if (m) return { name: m[1], head: parseInt(m[2], 10) };
  return { name: key, head: null };
}

// ─── 各行列の式展開 ────────────────────────────────────────

function renderThreeStep(title, s1, s2, s3, note = "") {
  return `
    <div class="explain-section">
      <h4>${escapeHtml(title)}</h4>
      <h4 class="step-h">[段1] 一般形</h4>${s1}
      <h4 class="step-h">[段2] 当てはめ</h4>${s2}
      <h4 class="step-h">[段3] 数値</h4>${s3}
      ${note}
    </div>`;
}

// W_E[token_id, d] のセル — token 埋め込み (位置に依存しない)
function renderWE(net, sel) {
  const { row: t, col: d } = sel;  // ここでの row は (現在表示中の) token 位置
  const tokenId = net.tokens[t];
  const word = net.vocab[tokenId];
  const we = net.W_E[tokenId * net.d_model + d];

  const s1 = mblock(mline(
    `${mv("W_E", "id,d")} <span class="comment">  (id = ${escapeHtml(word)} の vocab id)</span>`
  ));
  const s2 = mblock(mline(
    `${mv("W_E")}${mo("[")}id=${tokenId} (${escapeHtml(word)})${mo(",")} d${d}${mo("]")}`
  ));
  const s3 = mblock(mline(
    `${mv("W_E")}${mo("[")}id=${tokenId}${mo(",")} d${d}${mo("]")} ${mo("=")} ${mn(fmt(we))}`
  ));
  return renderThreeStep(
    `W_E (token 埋め込み) — 位置とは無関係`,
    s1, s2, s3,
    `<p class="muted note">※ W_E は preset で hand-crafted (=「猫」は dim 0 (is_noun) と dim 9 (サイズ特徴) に値を持つ、など意味が割り当ててある)。設計書 §6.9.2 の表を参照。同じ token は位置 t によらず常に同じ W_E ベクトルを返す。</p>`,
  );
}

// W_P[t, d] のセル — 位置エンコーディング (token と無関係)
function renderWP(net, sel) {
  const { row: t, col: d } = sel;
  const wp = net.W_P[t * net.d_model + d];
  const isEven = (d % 2 === 0);
  const i = Math.floor(d / 2);          // 偶数 dim 2i / 奇数 dim 2i+1 の i
  const power = (2 * i) / net.d_model;  // 10000^(2i/d_model) の指数
  const base = Math.pow(10000, power);
  const theta = t / base;
  const fnName = isEven ? "sin" : "cos";

  const s1 = mblock(mline(
    isEven
      ? `${mv("W_P", "t,2i")} ${mo("=")} ${mv("sin")}${mo("(")}t ${mo("/")} 10000<sup>2i/d_model</sup>${mo(")")}`
      : `${mv("W_P", "t,2i+1")} ${mo("=")} ${mv("cos")}${mo("(")}t ${mo("/")} 10000<sup>2i/d_model</sup>${mo(")")}`
  ));
  const s2 = mblock(mline(
    `${mv("W_P")}${mo("[")}t=${t}${mo(",")} d=${d}${mo("]")} ${mo("=")} ${mv(fnName)}${mo("(")}` +
    `${t} ${mo("/")} 10000<sup>${(2 * i)}/${net.d_model}</sup>${mo(")")}`
  ));
  const s3 = mblock(
    mline(`10000<sup>${(2 * i)}/${net.d_model}</sup> ${mo("=")} ${mn(base.toFixed(4))}`),
    mline(`${t} ${mo("/")} ${mn(base.toFixed(4))} ${mo("=")} ${mn(theta.toFixed(4))}`),
    mline(`${mv(fnName)}${mo("(")}${mn(theta.toFixed(4))}${mo(")")} ${mo("=")} ${mn(fmt(wp))}`),
  );
  return renderThreeStep(
    `W_P (位置エンコーディング) — token と無関係に位置 t だけで決まる`,
    s1, s2, s3,
    `<p class="muted note">※ 偶数 dim には sin、奇数 dim には cos を使う。dim が大きいほど周期が長い (10000<sup>2i/d_model</sup> が大きくなる)。学習せず固定値。同じ位置 t なら、どの token のときでも W_P は完全に同じ値。</p>`,
  );
}

// W_Q / W_K / W_V[in d, out k] のセル — 射影行列の 1 要素。
// 行 i = 入力 dim (X 側)、列 j = 出力 dim (Q/K/V 側)。preset で hand-crafted。
function renderProjectionWeight(net, sel, which) {
  const { row: inD, col: outG } = sel;
  const HD = net.h * net.d_k;
  const W = (which === "Q") ? net.blocks[0].W_Q
          : (which === "K") ? net.blocks[0].W_K
          :                   net.blocks[0].W_V;
  const w = W[inD * HD + outG];
  const matName = `W_${which}`;
  const outName = which;
  const hi = Math.floor(outG / net.d_k);
  const localK = outG % net.d_k;

  const dimMeaning = INPUT_DIM_MEANING[inD] ?? `dim ${inD}`;
  const bondMeaning = BOND_MEANING[outG] ?? `head ${hi} の col ${localK}`;

  const s1 = mblock(mline(
    `${mv(matName, "inD, outG")} <span class="comment">  (= 入力 dim inD の値を、出力 col outG = h${hi}.d${localK} にどれだけ流すか)</span>`
  ));
  const s2 = mblock(mline(
    `${mv(matName)}${mo("[")}d${inD}${mo(",")} g${outG}${mo("]")} <span class="comment">  (g${outG} = h${hi}·d_k + d${localK})</span>`
  ));
  const s3 = mblock(mline(
    `${mv(matName)}${mo("[")}d${inD}${mo(",")} g${outG}${mo("]")} ${mo("=")} ${mn(fmt(w))}`
  ));

  let note = "";
  if (which === "V") {
    note = `<p class="muted note">※ 本シミュレータの W_V は単位行列 (対角だけ +1.000、他は 0)。つまり V = X となり、Value 経路は「埋め込みベクトルそのままを後段に渡す」教育的設計を採っている。</p>`;
  } else {
    const role = (which === "Q")
      ? "Query (= 各 token が「どんな相手を見たいか」の問い合わせ) を作る"
      : "Key   (= 各 token が「自分はこう参照される」と差し出すラベル) を作る";
    const interp = (Math.abs(w) > 1e-9)
      ? `→ X[t, d${inD}] (= ${escapeHtml(dimMeaning)}) が ${outName}[t, h${hi}, d${localK}] (= ${escapeHtml(bondMeaning)}) に係数 ${fmt(w)} で流れ込む。`
      : `→ X[t, d${inD}] (= ${escapeHtml(dimMeaning)}) は ${outName}[t, h${hi}, d${localK}] には寄与しない (このパスは preset で閉じている)。`;
    note = `<p class="muted note">※ ${escapeHtml(matName)} は X を ${role} 重み行列。preset では「各文法カテゴリの dim」を「特定の bond の global col」に流すよう hand-crafted。${interp}</p>`;
  }

  return renderThreeStep(
    `${matName} (X → ${outName} 射影の 1 要素) — preset 由来の固定値`,
    s1, s2, s3,
    note,
  );
}

// preset §6.9 の dim 設計 (W_Q/W_K の解釈用)
const INPUT_DIM_MEANING = {
  0: "is_noun (名詞カテゴリ)",
  1: "is_adjective (形容詞カテゴリ)",
  2: "is_pronoun (指示詞カテゴリ)",
  3: "is_predicate (述語カテゴリ)",
  8: "美しさ特徴",
  9: "サイズ特徴",
  10: "書物特徴",
  11: "人気度特徴",
};
// bond は Q → K の一方向。Tier 2 では bond は global col で識別。
// bond 0/1 = head 0 (cols 0,1)、bond 2 = head 1 (col 8)。
const BOND_MEANING = {
  0: "bond 0 = 形容詞 (Q) → 名詞 (K)  [head 0]",
  1: "bond 1 = 指示詞 (Q) → 名詞 (K)  [head 0]",
  8: "bond 2 = 述語 (Q) → 名詞 (K)  [head 1]",
};

// X[t, d] = W_E[token[t], d] + W_P[t, d]
function renderEmbed(net, sel) {
  const { row: t, col: d } = sel;
  const tokenId = net.tokens[t];
  const word = net.vocab[tokenId];
  const we = net.W_E[tokenId * net.d_model + d];
  const wp = net.W_P[t * net.d_model + d];
  const x = net.X[t * net.d_model + d];

  const s1 = mblock(mline(
    `${mv("X", "t,d")} ${mo("=")} ${mv("W_E", "token[t],d")} ${mo("+")} ${mv("W_P", "t,d")}`
  ));
  const s2 = mblock(mline(
    `${mv("X")}${mo("[")}${tokenLabel(net, t)}${mo(",")} d${d}${mo("]")} ${mo("=")} ` +
    `${mv("W_E")}${mo("[")}id=${tokenId} (${word})${mo(",")} d${d}${mo("]")} ${mo("+")} ` +
    `${mv("W_P")}${mo("[")}t=${t}${mo(",")} d${d}${mo("]")}`
  ));
  const s3 = mblock(mline(
    `${mn(fmt(we))} ${mo("+")} ${mn(fmt(wp))} ${mo("=")} ${mn(fmt(x))}`
  ));
  return renderThreeStep(
    `Embed (X) — トークン埋め込み + 位置エンコーディング`,
    s1, s2, s3,
    `<p class="muted note">※ W_E は token id ごとの埋め込み (hand-crafted)、W_P は固定の sin/cos 位置エンコーディング。</p>`,
  );
}

// Q[t, hi, k] = sum_d X[t, d] · W_?[d, hi*d_k + k]   (Tier 2: Multi-Head)
// view.js は head ごとに (T, d_k) を表示するので、sel.col は head 内ローカル col k。
// 重み W_? の global col は hi * d_k + k (= W のグローバル列 g) になる。
function renderQKV(net, sel, which, headIdx) {
  const { row: t, col: k } = sel;
  const D = net.d_model;
  const d_k = net.d_k;
  const h = net.h;
  const hi = (headIdx == null) ? 0 : headIdx;
  const g = hi * d_k + k;  // global col index (W_Q / Q_full の列)
  const W = (which === "Q") ? net.blocks[0].W_Q
          : (which === "K") ? net.blocks[0].W_K
          : net.blocks[0].W_V;
  const out = (which === "Q") ? net.Q
            : (which === "K") ? net.K
            : net.V;
  const v = out[t * h * d_k + hi * d_k + k];

  const s1 = mblock(mline(
    `${mv(which, "t,hi,k")} ${mo("=")} ${mv("Σ", "d")} ${mv("X", "t,d")} ${mo("·")} ` +
    `${mv(`W_${which}`, "d, hi·d_k+k")}`
  ));
  const s2 = mblock(mline(
    `${mv(which)}${mo("[")}${tokenLabel(net, t)}${mo(",")} h${hi}${mo(",")} d${k}${mo("]")} ${mo("=")} ` +
    `${mv("Σ", "d")} ${mv("X")}${mo("[")}t=${t}${mo(",")} d${mo("]")} ${mo("·")} ` +
    `${mv(`W_${which}`)}${mo("[")}d${mo(",")} g=${g}${mo("]")} <span class="comment">  (g = h${hi}·d_k + d${k} = ${g})</span>`
  ));

  // [段3] 数値展開: 0 でない項のみ抽出
  const lines = [];
  let total = 0;
  for (let d = 0; d < D; d++) {
    const x  = net.X[t * D + d];
    const wd = W[d * h * d_k + g];
    const prod = x * wd;
    total += prod;
    if (Math.abs(wd) < 1e-9) continue;
    lines.push(
      `${mv("X")}${mo("[")}t,${d}${mo("]")} ${mo("·")} ${mv(`W_${which}`)}${mo("[")}${d},g${g}${mo("]")} ${mo("=")} ` +
      `${mn(fmt(x))} ${mo("·")} ${mn(fmt(wd))} ${mo("=")} ${mn(fmt(prod))}`
    );
  }
  if (lines.length === 0) {
    lines.push(
      `<span class="comment"># すべての項で W_${which}[d, g${g}] = 0 のため寄与なし</span>`
    );
  }
  lines.push(
    `${mv(which)}${mo("[")}${tokenLabel(net, t)}${mo(",")} h${hi}${mo(",")} d${k}${mo("]")} ${mo("=")} ${mn(fmt(total))}`
  );
  const s3 = mblock(...lines.map(mline));

  const intent = which === "Q" ? "Query (このトークンが「何を見たいか」)"
              : which === "K" ? "Key   (このトークンが「何として参照されるか」)"
              : "Value (このトークンが「何を持ち寄るか」)";
  return renderThreeStep(
    `${which} — ${intent}   [head ${hi}]`,
    s1, s2, s3,
    `<p class="muted note">※ W_${which} は preset で hand-crafted。head ${hi} に割り当てられた cols (global g = ${hi * d_k}..${hi * d_k + d_k - 1}) のうち、bond 配線が立っている col だけ非ゼロが出る。</p>`,
  );
}

// scores[i, j] = (Q[i, :] · K[j, :]) / sqrt(d_k)
function renderScores(net, sel, headIdx) {
  const { row: i, col: j } = sel;
  const T = net.T;
  const d_k = net.d_k;
  const h = net.h;
  const hi = (headIdx == null) ? 0 : headIdx;
  const score = net.scores[hi * T * T + i * T + j];
  const sqrtDk = Math.sqrt(d_k);

  const s1 = mblock(mline(
    `${mv("scores", "hi,i,j")} ${mo("=")} ${mo("(")}${mv("Q", "i,hi,:")} ${mo("·")} ${mv("K", "j,hi,:")}${mo(")")} ${mo("/")} ${mv("√d_k")}`
  ));
  const s2 = mblock(mline(
    `${mv("scores")}${mo("[")}h${hi}, ${tokenLabel(net, i)} → ${tokenLabel(net, j)}${mo("]")} ${mo("=")} ` +
    `${mo("(")}${mv("Σ", "k")} ${mv("Q")}${mo("[")}i,h${hi},k${mo("]")} ${mo("·")} ` +
    `${mv("K")}${mo("[")}j,h${hi},k${mo("]")}${mo(")")} ${mo("/")} ${mn(sqrtDk.toFixed(3))}`
  ));

  // [段3]: 0 でない項を出す。head 内 (T, d_k) のオフセット計算
  const qBase = i * h * d_k + hi * d_k;
  const kBase = j * h * d_k + hi * d_k;
  const lines = [];
  let dot = 0;
  for (let k = 0; k < d_k; k++) {
    const q = net.Q[qBase + k];
    const kk = net.K[kBase + k];
    const prod = q * kk;
    dot += prod;
    if (Math.abs(prod) < 1e-9) continue;
    lines.push(
      `${mv("Q")}${mo("[i,h")}${hi}${mo(",")}${k}${mo("]")} ${mo("·")} ${mv("K")}${mo("[j,h")}${hi}${mo(",")}${k}${mo("]")} ${mo("=")} ` +
      `${mn(fmt(q))} ${mo("·")} ${mn(fmt(kk))} ${mo("=")} ${mn(fmt(prod))}`
    );
  }
  if (lines.length === 0) {
    lines.push(`<span class="comment"># Q · K のすべての項で 0 (= score = 0)</span>`);
  }
  lines.push(
    `${mv("Q")} ${mo("·")} ${mv("K")} ${mo("=")} ${mn(fmt(dot))}`
  );
  lines.push(
    `${mv("scores")} ${mo("=")} ${mn(fmt(dot))} ${mo("/")} ${mn(sqrtDk.toFixed(3))} ${mo("=")} ${mn(fmt(score))}`
  );
  const s3 = mblock(...lines.map(mline));

  const qkExcerpt = renderQKExcerpt(net, i, j, hi);

  return renderThreeStep(
    `scores — Attention スコア (内積 / √d_k)   [head ${hi}]`,
    s1, s2, s3,
    qkExcerpt +
    `<p class="muted note">※ √d_k = √${d_k} で割るのは d_k が大きいときに softmax が極端化しないためのスケーリング (元論文 §3.2.1)。Multi-Head の場合は各 head の d_k で割る。</p>`,
  );
}

// scores[hi, i, j] のセルクリック時に出す「Q[i, hi] と K[j, hi] の関連 dim 抜粋」表。
function renderQKExcerpt(net, i, j, hi) {
  const d_k = net.d_k;
  const h = net.h;
  const qBase = i * h * d_k + hi * d_k;
  const kBase = j * h * d_k + hi * d_k;
  const cols = [];
  for (let k = 0; k < d_k; k++) {
    const q = net.Q[qBase + k];
    const kk = net.K[kBase + k];
    if (Math.abs(q) > 1e-9 || Math.abs(kk) > 1e-9) {
      cols.push({ k, q, kk, prod: q * kk });
    }
  }
  if (cols.length === 0) {
    return `<p class="muted note">※ 関連 dim 抜粋: Q[${tokenLabel(net, i)}, h${hi}] も K[${tokenLabel(net, j)}, h${hi}] も全 dim で 0 のため、scores も 0。</p>`;
  }
  const head = `<th>dim</th>` + cols.map(c => `<th>h${hi}.d${c.k}</th>`).join("");
  const qRow = `<th>Q[i]</th>` + cols.map(c => `<td>${fmt(c.q)}</td>`).join("");
  const kRow = `<th>K[j]</th>` + cols.map(c => `<td>${fmt(c.kk)}</td>`).join("");
  const pRow = `<th>Q·K</th>` + cols.map(c => `<td>${fmt(c.prod)}</td>`).join("");
  return `
    <div class="qk-excerpt">
      <p class="qk-excerpt-cap">関連セル抜粋: head ${hi}, i = ${escapeHtml(tokenLabel(net, i))}, j = ${escapeHtml(tokenLabel(net, j))} (Q または K が 0 でない dim だけ)</p>
      <table class="qk-pair-table">
        <thead><tr>${head}</tr></thead>
        <tbody>
          <tr>${qRow}</tr>
          <tr>${kRow}</tr>
          <tr class="qk-prod-row">${pRow}</tr>
        </tbody>
      </table>
    </div>`;
}

// attn[hi, i, j] = softmax(scores[hi, i, :])[j]   (Tier 2: per-head softmax)
function renderAttn(net, sel, headIdx) {
  const { row: i, col: j } = sel;
  const T = net.T;
  const hi = (headIdx == null) ? 0 : headIdx;
  const sBase = hi * T * T;
  const a = net.attn[sBase + i * T + j];
  const score_ij = net.scores[sBase + i * T + j];

  const s1 = mblock(mline(
    `${mv("attn", "hi,i,j")} ${mo("=")} ${mv("softmax")}${mo("(")}${mv("scores", "hi,i,:")}${mo(")")}${mo("[")}j${mo("]")}`
  ));
  const s2 = mblock(mline(
    `${mv("attn")}${mo("[")}h${hi}, ${tokenLabel(net, i)} → ${tokenLabel(net, j)}${mo("]")} ` +
    `${mo("=")} ${mv("e")}<sup>${fmt(score_ij)}</sup> ${mo("/")} ` +
    `${mv("Σ", "j'")} ${mv("e")}<sup>scores[h${hi}, i, j']</sup>`
  ));

  // [段3] 数値: 各 j' について exp(score) と総和 (head 内のみ)
  let m = -Infinity;
  for (let jp = 0; jp < T; jp++) {
    const s = net.scores[sBase + i * T + jp];
    if (s > m) m = s;
  }
  let sum = 0;
  const lines3 = [];
  for (let jp = 0; jp < T; jp++) {
    const s = net.scores[sBase + i * T + jp];
    const e = Math.exp(s - m);
    sum += e;
    lines3.push(
      `${mv("e")}<sup>${fmt(s)} − max</sup> ` +
      `${mo("=")} ${mn(e.toFixed(4))} <span class="comment">  (${tokenLabel(net, jp)} に対する重み素材)</span>`
    );
  }
  lines3.push(
    `${mv("Σ")} ${mo("=")} ${mn(sum.toFixed(4))}`
  );
  lines3.push(
    `${mv("attn")}${mo("[")}h${hi}, i, j${mo("]")} ${mo("=")} ${mv("e")}<sup>${fmt(score_ij)} − max</sup> / ${mn(sum.toFixed(4))} ${mo("=")} ${mn(fmtAttn(a, 4))}`
  );
  const s3 = mblock(...lines3.map(mline));

  return renderThreeStep(
    `attn — Softmax 後の注目度   [head ${hi}]   (各行の和 = 1)`,
    s1, s2, s3,
    `<p class="muted note">※ 数値安定化のため、各 head の各行の max を引いてから exp を計算 (overflow 防止)。値の比率は不変。Multi-Head の softmax は head 内で独立に行う。</p>`,
  );
}

// attnOut[t, hi, k] = sum_j(attn[hi, t, j] · V[j, hi, k])   (Tier 2: per-head)
function renderAttnOut(net, sel, headIdx) {
  const { row: i, col: k } = sel;
  const T = net.T;
  const d_k = net.d_k;
  const h = net.h;
  const hi = (headIdx == null) ? 0 : headIdx;
  const v = net.attnOut[i * h * d_k + hi * d_k + k];
  const aBase = hi * T * T + i * T;

  const s1 = mblock(mline(
    `${mv("attnOut", "t,hi,k")} ${mo("=")} ${mv("Σ", "j")} ${mv("attn", "hi,t,j")} ${mo("·")} ${mv("V", "j,hi,k")}`
  ));
  const s2 = mblock(mline(
    `${mv("attnOut")}${mo("[")}${tokenLabel(net, i)}, h${hi}, d${k}${mo("]")} ${mo("=")} ` +
    `${mv("Σ", "j")} ${mv("attn")}${mo("[")}h${hi}, i, j${mo("]")} ${mo("·")} ${mv("V")}${mo("[")}j, h${hi}, ${k}${mo("]")}`
  ));

  const lines = [];
  let total = 0;
  for (let j = 0; j < T; j++) {
    const a = net.attn[aBase + j];
    const vj = net.V[j * h * d_k + hi * d_k + k];
    const prod = a * vj;
    total += prod;
    lines.push(
      `${mn(fmtAttn(a, 3))} ${mo("·")} ${mn(fmt(vj))} ${mo("=")} ${mn(fmt(prod))} ` +
      `<span class="comment">  (${tokenLabel(net, j)} の V から)</span>`
    );
  }
  lines.push(
    `${mv("attnOut")}${mo("[")}t, h${hi}, k${mo("]")} ${mo("=")} ${mn(fmt(total))}`
  );
  const s3 = mblock(...lines.map(mline));

  return renderThreeStep(
    `attnOut — Attention で集約した V   [head ${hi}]`,
    s1, s2, s3,
    `<p class="muted note">※ 各 j について attn と V の積を合計。attn が大きい j (= 注目しているトークン) の V の寄与が大きくなる。各 head 独立に集約し、最後に concat して W_O で射影する。</p>`,
  );
}

// Y[i, d] = sum_g concat(attnOut)[i, g] · W_O[g, d]   (Tier 2: g は h*d_k 軸)
function renderY(net, sel) {
  const { row: i, col: d } = sel;
  const HD = net.h * net.d_k;  // = d_model
  const W_O = net.blocks[0].W_O;
  const y = net.Y[i * net.d_model + d];

  const s1 = mblock(mline(
    `${mv("Y", "i,d")} ${mo("=")} ${mv("Σ", "g")} ${mv("attnOut_concat", "i,g")} ${mo("·")} ${mv("W_O", "g,d")} ` +
    `<span class="comment">  (g は head 軸を flatten した index)</span>`
  ));
  const s2 = mblock(mline(
    `${mv("Y")}${mo("[")}${tokenLabel(net, i)}${mo(",")} d${d}${mo("]")} ${mo("=")} ` +
    `${mv("Σ", "g")} ${mv("attnOut_concat")}${mo("[")}i, g${mo("]")} ${mo("·")} ${mv("W_O")}${mo("[")}g, ${d}${mo("]")}`
  ));

  const lines = [];
  let total = 0;
  // attnOut の (T, h*d_k) 配置は net.attnOut[i * HD + g] でアクセスできる。
  for (let g = 0; g < HD; g++) {
    const a = net.attnOut[i * HD + g];
    const w = W_O[g * net.d_model + d];
    const prod = a * w;
    total += prod;
    if (Math.abs(w) < 1e-9) continue;
    const hi = Math.floor(g / net.d_k);
    const k  = g % net.d_k;
    lines.push(
      `${mn(fmt(a))} ${mo("·")} ${mn(fmt(w))} ${mo("=")} ${mn(fmt(prod))} ` +
      `<span class="comment">  (g${g} = h${hi}.d${k})</span>`
    );
  }
  if (lines.length === 0) {
    lines.push(`<span class="comment"># W_O[g,d] が全て 0 のため寄与なし</span>`);
  }
  lines.push(`${mv("Y")} ${mo("=")} ${mn(fmt(total))}`);
  const s3 = mblock(...lines.map(mline));

  return renderThreeStep(
    `Y — Multi-Head Attention の射影出力 (concat · W_O)`,
    s1, s2, s3,
    `<p class="muted note">※ 本シミュレータでは W_O = 単位行列なので、Y = concat(attnOut)。各 head の出力がそのまま並ぶ identity 設計を hand-crafted で維持してある。</p>`,
  );
}

// residual1[t, d] = X[t, d] + Y[t, d]   (残差接続: 元の token 情報を attention 出力に足し戻す)
function renderResidual1(net, sel) {
  const { row: t, col: d } = sel;
  const D = net.d_model;
  const x = net.X[t * D + d];
  const y = net.Y[t * D + d];
  const r = net.residual1[t * D + d];

  const s1 = mblock(mline(
    `${mv("residual1", "t,d")} ${mo("=")} ${mv("X", "t,d")} ${mo("+")} ${mv("Y", "t,d")}`
  ));
  const s2 = mblock(mline(
    `${mv("residual1")}${mo("[")}${tokenLabel(net, t)}${mo(",")} d${d}${mo("]")} ${mo("=")} ` +
    `${mv("X")}${mo("[")}${tokenLabel(net, t)}${mo(",")} d${d}${mo("]")} ${mo("+")} ` +
    `${mv("Y")}${mo("[")}${tokenLabel(net, t)}${mo(",")} d${d}${mo("]")}`
  ));
  const s3 = mblock(mline(
    `${mn(fmt(x))} ${mo("+")} ${mn(fmt(y))} ${mo("=")} ${mn(fmt(r))}`
  ));

  return renderThreeStep(
    `residual1 — 残差接続 (元 token 情報 X を Multi-Head 出力 Y に足し戻す)`,
    s1, s2, s3,
    `<p class="muted note">※ 残差接続 (residual connection) の役目: attention で集めた情報 Y だけだと元の token 情報が失われがちなので、X をそのまま足し戻すことで「元の自分 + 周りから集めた情報」の合算ベクトルにする。深い Transformer での勾配消失を防ぐ働きもある。</p>`,
  );
}

// ln1_out[t, d] = γ_d * (residual1[t, d] - μ_t) / sqrt(σ²_t + ε) + β_d
function renderLN1(net, sel) {
  const { row: t, col: d } = sel;
  const D = net.d_model;
  const block0 = net.blocks[0];
  const gamma = block0.LN1_gamma[d];
  const beta  = block0.LN1_beta[d];
  // μ, σ² を行 t について計算
  let mu = 0;
  for (let dd = 0; dd < D; dd++) mu += net.residual1[t * D + dd];
  mu /= D;
  let var_ = 0;
  for (let dd = 0; dd < D; dd++) {
    const dx = net.residual1[t * D + dd] - mu;
    var_ += dx * dx;
  }
  var_ /= D;
  const eps = 1e-5;
  const sigma = Math.sqrt(var_ + eps);
  const r = net.residual1[t * D + d];
  const x_norm = (r - mu) / sigma;
  const y_out = gamma * x_norm + beta;

  const s1 = mblock(mline(
    `${mv("ln1_out", "t,d")} ${mo("=")} ` +
    `${mv("γ", "d")} ${mo("·")} ${mo("(")}${mv("residual1", "t,d")} ${mo("-")} ${mv("μ", "t")}${mo(")")} ${mo("/")} ` +
    `${mv("√(σ²")}<sub>t</sub>${mv(" + ε)")} ${mo("+")} ${mv("β", "d")}`
  ));
  const s2 = mblock(
    mline(`${mv("μ", "t")} ${mo("=")} (1 / d_model) ${mv("Σ", "d'")} ${mv("residual1")}${mo("[")}${tokenLabel(net, t)}, d'${mo("]")} ${mo("=")} ${mn(mu.toFixed(4))}`),
    mline(`${mv("σ²", "t")} ${mo("=")} (1 / d_model) ${mv("Σ", "d'")} ${mo("(")}${mv("residual1")}${mo("[")}${tokenLabel(net, t)}, d'${mo("]")} ${mo("-")} ${mv("μ", "t")}${mo(")²")} ${mo("=")} ${mn(var_.toFixed(4))}`),
    mline(`${mv("√(σ²")}<sub>t</sub>${mv(" + ε)")} ${mo("=")} ${mn(sigma.toFixed(4))} <span class="comment">  (ε = 1e-5 で 0 除算回避)</span>`),
  );
  const s3 = mblock(
    mline(`${mv("residual1")}${mo("[")}${tokenLabel(net, t)}, d${d}${mo("]")} ${mo("-")} ${mv("μ")} ${mo("=")} ${mn(fmt(r))} ${mo("-")} ${mn(mu.toFixed(4))} ${mo("=")} ${mn((r - mu).toFixed(4))}`),
    mline(`${mo("(")}${mn((r - mu).toFixed(4))}${mo(")")} ${mo("/")} ${mn(sigma.toFixed(4))} ${mo("=")} ${mn(x_norm.toFixed(4))} <span class="comment">  (標準化された値)</span>`),
    mline(`${mv("γ")}<sub>${d}</sub> ${mo("·")} ${mn(x_norm.toFixed(4))} ${mo("+")} ${mv("β")}<sub>${d}</sub> ${mo("=")} ${mn(gamma.toFixed(3))} ${mo("·")} ${mn(x_norm.toFixed(4))} ${mo("+")} ${mn(beta.toFixed(3))} ${mo("=")} ${mn(fmt(y_out))}`),
  );

  return renderThreeStep(
    `ln1_out — Layer Normalization (per-token、d_model 軸方向の正規化)`,
    s1, s2, s3,
    `<p class="muted note">※ LayerNorm は各 token (= 各行) について d_model 軸の平均 μ と分散 σ² を取り、(x - μ) / √(σ² + ε) で標準化したあと γ・β でスケール/シフトする。本シミュレータの γ = 1, β = 0 では純粋な「平均 0、標準偏差 1」への正規化になり、token ごとに値の分布の大きさが揃う。実 Transformer では γ, β も学習で「適切なスケール」を獲得する。</p>`,
  );
}

// ─── FFN (Feed-Forward Network、2 層 MLP + GELU) ──────────

// preset の FFN 検出器ニューロン意図 (build_preset.py の FFN_NEURON_INTENT と同期)
const FFN_NEURON_INTENT = {
  0: "名詞 dim (d0) 検出器",
  1: "形容詞 dim (d1) 検出器",
  2: "指示詞 dim (d2) 検出器",
  3: "述語 dim (d3) 検出器",
  4: "助詞「は」(d4) 検出器",
  5: "助詞「が」(d5) 検出器",
  6: "助詞「を」(d6) 検出器",
  7: "コピュラ「です」(d7) 検出器",
  8: "美しさ特徴 (d8) 検出器",
  9: "サイズ特徴 (d9) 検出器",
  10: "書物特徴 (d10) 検出器",
  11: "人気度特徴 (d11) 検出器",
  12: "AND: 名詞 ∧ 美しさ → 美しい名詞検出 (d8 を強化)",
  13: "AND: 名詞 ∧ サイズ → サイズを持つ名詞検出 (d9 を強化)",
  14: "AND: 名詞 ∧ 書物 → 書物検出 (d10 を強化)",
  15: "AND: 名詞 ∧ 人気度 → 人気な名詞検出 (d11 を強化)",
};

// W1 / W2 のセル — preset 由来の固定重み
function renderFFNWeight(net, sel, which) {
  const { row, col } = sel;
  const block0 = net.blocks[0];
  const W = which === "W1" ? block0.FFN_W1 : block0.FFN_W2;
  const matName = `FFN_${which}`;
  const inLabel = which === "W1" ? `d${row}` : `h${row}`;
  const outLabel = which === "W1" ? `h${col}` : `d${col}`;
  const cols = which === "W1" ? net.d_ff : net.d_model;
  const w = W[row * cols + col];

  const s1 = mblock(mline(
    `${mv(matName, "in,out")} <span class="comment">  (= 入力 ${inLabel} の値を、出力 ${outLabel} にどれだけ流すか)</span>`
  ));
  const s2 = mblock(mline(
    `${mv(matName)}${mo("[")}${inLabel}${mo(",")} ${outLabel}${mo("]")}`
  ));
  const s3 = mblock(mline(
    `${mv(matName)}${mo("[")}${inLabel}${mo(",")} ${outLabel}${mo("]")} ${mo("=")} ${mn(fmt(w))}`
  ));

  let note = "";
  if (which === "W1") {
    const hk = col;
    const intent = FFN_NEURON_INTENT[hk];
    if (intent) {
      note = `<p class="muted note">※ FFN_W1 の列 h${hk} は preset の検出器ニューロン: <b>${escapeHtml(intent)}</b>。各 token の ln1_out から該当 dim を「立っているか」判定する閾値ロジックを実現。</p>`;
    } else {
      note = `<p class="muted note">※ FFN_W1 の列 h${hk} は preset で未使用 (全 0)。</p>`;
    }
  } else {
    const hk = row;
    const intent = FFN_NEURON_INTENT[hk];
    if (intent) {
      note = `<p class="muted note">※ FFN_W2 の行 h${hk} (検出器: ${escapeHtml(intent)}) は、その活性を出力 d_model のどの dim に書き戻すかを定める。検出器の発火を「特定 dim の値強化」として後段に伝える。</p>`;
    } else {
      note = `<p class="muted note">※ h${hk} は preset で未使用なので、この行はすべて 0。</p>`;
    }
  }

  return renderThreeStep(
    `${matName} — FFN の射影重み (preset で hand-crafted)`,
    s1, s2, s3,
    note,
  );
}

// ffn_h[t, k] = GELU(Σ_d ln1_out[t, d] · W1[d, k] + b1[k])
function renderFFNHidden(net, sel) {
  const { row: t, col: k } = sel;
  const D = net.d_model;
  const block0 = net.blocks[0];
  const b1 = block0.FFN_b1[k];
  const intent = FFN_NEURON_INTENT[k] ?? `h${k} (preset で未使用)`;

  // 0 でない項のみ展開
  let z = b1;
  const lines2 = [];
  for (let d = 0; d < D; d++) {
    const w = block0.FFN_W1[d * net.d_ff + k];
    if (Math.abs(w) < 1e-9) continue;
    const x = net.ln1_out[t * D + d];
    z += x * w;
    lines2.push(
      `${mv("ln1_out")}${mo("[")}${tokenLabel(net, t)}, d${d}${mo("]")} ${mo("·")} ${mv("W1")}${mo("[")}d${d}, h${k}${mo("]")} ${mo("=")} ` +
      `${mn(fmt(x))} ${mo("·")} ${mn(fmt(w))} ${mo("=")} ${mn(fmt(x * w))}`
    );
  }
  if (lines2.length === 0) {
    lines2.push(`<span class="comment"># W1[d, h${k}] が全 d で 0 (= 検出器が preset で未使用)</span>`);
  }
  // GELU 計算
  const GELU_COEF = Math.sqrt(2 / Math.PI);
  const inner = GELU_COEF * (z + 0.044715 * z * z * z);
  const ge = 0.5 * z * (1 + Math.tanh(inner));

  const s1 = mblock(mline(
    `${mv("ffn_h", "t,k")} ${mo("=")} ${mv("GELU")}${mo("(")}${mv("Σ", "d")} ${mv("ln1_out", "t,d")} ${mo("·")} ${mv("W1", "d,k")} ${mo("+")} ${mv("b1", "k")}${mo(")")}`
  ));
  const s2 = mblock(...lines2.map(mline),
    mline(`${mv("b1")}${mo("[")}h${k}${mo("]")} ${mo("=")} ${mn(fmt(b1))}`),
    mline(`${mv("Σ + b1")} ${mo("=")} ${mn(fmt(z))} <span class="comment">  (= GELU 入力 ffn_pre[t, h${k}])</span>`),
  );
  const s3 = mblock(mline(
    `${mv("GELU")}${mo("(")}${mn(fmt(z))}${mo(")")} ${mo("=")} ${mn(fmt(ge))}`
  ));

  return renderThreeStep(
    `ffn_h — FFN 中間層の検出器活性   [h${k}: ${intent}]`,
    s1, s2, s3,
    `<p class="muted note">※ GELU は ReLU と似た滑らかな活性化で、入力が大きく正なら ≈ x、負なら ≈ 0 を返す。本 preset では W1 を「特定 dim 同時発火検出器」として作ってあるので、ffn_h[t, k] が正の大きな値 = 「token t において k 番ニューロンが捉えた特徴が立っている」と読める。</p>`,
  );
}

// ffn_out[t, d] = Σ_k ffn_h[t, k] · W2[k, d] + b2[d]
function renderFFNOut(net, sel) {
  const { row: t, col: d } = sel;
  const block0 = net.blocks[0];
  const d_ff = net.d_ff;
  const b2 = block0.FFN_b2[d];

  let total = b2;
  const lines = [];
  for (let k = 0; k < d_ff; k++) {
    const w = block0.FFN_W2[k * net.d_model + d];
    if (Math.abs(w) < 1e-9) continue;
    const a = net.ffn_h[t * d_ff + k];
    const prod = a * w;
    total += prod;
    const intent = FFN_NEURON_INTENT[k] ?? `h${k}`;
    lines.push(
      `${mv("ffn_h")}${mo("[")}t, h${k}${mo("]")} ${mo("·")} ${mv("W2")}${mo("[")}h${k}, d${d}${mo("]")} ${mo("=")} ` +
      `${mn(fmt(a))} ${mo("·")} ${mn(fmt(w))} ${mo("=")} ${mn(fmt(prod))} <span class="comment">  (h${k}: ${escapeHtml(intent)})</span>`
    );
  }
  if (lines.length === 0) {
    lines.push(`<span class="comment"># W2[*, d${d}] が全 0 のため寄与なし</span>`);
  }

  const s1 = mblock(mline(
    `${mv("ffn_out", "t,d")} ${mo("=")} ${mv("Σ", "k")} ${mv("ffn_h", "t,k")} ${mo("·")} ${mv("W2", "k,d")} ${mo("+")} ${mv("b2", "d")}`
  ));
  const s2 = mblock(...lines.map(mline),
    mline(`${mv("b2")}${mo("[")}d${d}${mo("]")} ${mo("=")} ${mn(fmt(b2))}`),
    mline(`${mv("ffn_out")}${mo("[")}${tokenLabel(net, t)}, d${d}${mo("]")} ${mo("=")} ${mn(fmt(total))}`),
  );
  const s3 = mblock(mline(
    `合計 = ${mn(fmt(total))}`
  ));

  return renderThreeStep(
    `ffn_out — FFN の出力 (検出器活性を d_model に書き戻し)`,
    s1, s2, s3,
    `<p class="muted note">※ W2 が「検出器の活性をどの dim に書き戻すか」を決めるので、発火した検出器に応じて出力 dim の特定 dim が強化される。本 preset では h12〜h15 の AND 検出器が「美しい名詞 → 美しさ +0.5」のように特徴 dim を上書き強化する。</p>`,
  );
}

// residual2[t, d] = ln1_out[t, d] + ffn_out[t, d]
function renderResidual2(net, sel) {
  const { row: t, col: d } = sel;
  const D = net.d_model;
  const a = net.ln1_out[t * D + d];
  const b = net.ffn_out[t * D + d];
  const r = net.residual2[t * D + d];

  const s1 = mblock(mline(
    `${mv("residual2", "t,d")} ${mo("=")} ${mv("ln1_out", "t,d")} ${mo("+")} ${mv("ffn_out", "t,d")}`
  ));
  const s2 = mblock(mline(
    `${mv("residual2")}${mo("[")}${tokenLabel(net, t)}, d${d}${mo("]")} ${mo("=")} ${mn(fmt(a))} ${mo("+")} ${mn(fmt(b))}`
  ));
  const s3 = mblock(mline(
    `${mv("residual2")} ${mo("=")} ${mn(fmt(r))}`
  ));

  return renderThreeStep(
    `residual2 — 2 回目の残差接続 (FFN 入力 + FFN 出力)`,
    s1, s2, s3,
    `<p class="muted note">※ FFN は「検出器が立った dim を強化」するだけの差分情報なので、元の ln1_out (= attention 部の出力) に足し戻して 2 回目の残差を作る。これにより attention で集めた情報も FFN で強化された情報もどちらも保持される。</p>`,
  );
}

// ln2_out[t, d] = LayerNorm(residual2)
function renderLN2(net, sel) {
  const { row: t, col: d } = sel;
  const D = net.d_model;
  const block0 = net.blocks[0];
  const gamma = block0.LN2_gamma[d];
  const beta  = block0.LN2_beta[d];
  let mu = 0;
  for (let dd = 0; dd < D; dd++) mu += net.residual2[t * D + dd];
  mu /= D;
  let var_ = 0;
  for (let dd = 0; dd < D; dd++) {
    const dx = net.residual2[t * D + dd] - mu;
    var_ += dx * dx;
  }
  var_ /= D;
  const eps = 1e-5;
  const sigma = Math.sqrt(var_ + eps);
  const r = net.residual2[t * D + d];
  const x_norm = (r - mu) / sigma;
  const y_out = gamma * x_norm + beta;

  const s1 = mblock(mline(
    `${mv("ln2_out", "t,d")} ${mo("=")} ${mv("γ", "d")} ${mo("·")} ${mo("(")}${mv("residual2", "t,d")} ${mo("-")} ${mv("μ", "t")}${mo(")")} ${mo("/")} ${mv("√(σ²")}<sub>t</sub>${mv(" + ε)")} ${mo("+")} ${mv("β", "d")}`
  ));
  const s2 = mblock(
    mline(`${mv("μ", "t")} ${mo("=")} ${mn(mu.toFixed(4))}, ${mv("σ²", "t")} ${mo("=")} ${mn(var_.toFixed(4))}, ${mv("√(σ²+ε)")} ${mo("=")} ${mn(sigma.toFixed(4))}`),
  );
  const s3 = mblock(
    mline(`(${mn(fmt(r))} ${mo("-")} ${mn(mu.toFixed(4))}) ${mo("/")} ${mn(sigma.toFixed(4))} ${mo("=")} ${mn(x_norm.toFixed(4))}`),
    mline(`${mv("γ")}<sub>${d}</sub> ${mo("·")} ${mn(x_norm.toFixed(4))} ${mo("+")} ${mv("β")}<sub>${d}</sub> ${mo("=")} ${mn(fmt(y_out))}`),
  );

  return renderThreeStep(
    `ln2_out — 2 回目の Layer Normalization (block 最終出力)`,
    s1, s2, s3,
    `<p class="muted note">※ ln2_out が 1 ブロック分の最終出力。次のブロック (本シミュレータでは N=1 なので無し) または最終予測層への入力になる。LN の効果は Lesson 9 と同じ: 各 token の dim 分布を平均 0・標準偏差 1 に揃える。</p>`,
  );
}

// ─── Attention Map (補助パネル常駐表示用、シンプルなテーブル HTML) ──

export function renderAttentionMapHtml(net) {
  if (net.phase === "idle") {
    return `<p class="muted">Forward 後、ここに attention map が常駐表示されます。</p>`;
  }
  const T = net.T;
  const H = net.h;
  // Multi-Head: head ごとに 1 つずつ table を縦に並べる。
  let html = "";
  for (let hi = 0; hi < H; hi++) {
    html += `<div class="attmap-head"><h4 class="attmap-head-title">head ${hi}</h4>`;
    html += `<table class="attmap-table"><thead><tr><th></th>`;
    for (let j = 0; j < T; j++) {
      html += `<th class="attmap-col">${escapeHtml(tokenLabel(net, j))}</th>`;
    }
    html += `</tr></thead><tbody>`;
    const sBase = hi * T * T;
    for (let i = 0; i < T; i++) {
      html += `<tr><th class="attmap-row">${escapeHtml(tokenLabel(net, i))}</th>`;
      for (let j = 0; j < T; j++) {
        const v = net.attn[sBase + i * T + j];
        const t = Math.max(0, Math.min(1, v));
        const r = Math.round(255 - (255 - 43) * t);
        const g = Math.round(255 - (255 - 108) * t);
        const b = Math.round(255 - (255 - 176) * t);
        const fg = v >= 0.5 ? "#fff" : "#222";
        html += `<td class="attmap-cell" style="background:rgb(${r},${g},${b});color:${fg}">${v.toFixed(2)}</td>`;
      }
      html += `</tr>`;
    }
    html += `</tbody></table></div>`;
  }
  return html;
}
