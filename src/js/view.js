// view.js — Tier 1 ネットワーク・ビューの SVG 描画 + pan/zoom。
//
// 設計書 §7 (可視化) に対応:
//   - 中央パネルに行列ヒートマップを並べる (行ヘッダ = 単語ラベル、列ヘッダ = dim 番号)
//   - タブ (Embed / Q-K-V / Attn / Out) で表示する行列セットを切り替える
//   - SVG viewBox を drag/wheel で pan/zoom できる
//
// インタラクション (セルクリック → 式の展開、ホバーでハイライト) は P5 で追加する。
// 本ファイルは「現状の net.X / Q / K / V / scores / attn / attnOut / Y を読んで描く」だけ。

import { row } from "./matrix.js";
import { headSliceTHD, headSliceHTT } from "./model.js";

// ─── レイアウト定数 ────────────────────────────────────────

const LAYOUT = {
  cellW: 50,         // セルの幅 (10 px の値「+0.412」が収まる)
  cellH: 22,         // セルの高さ
  rowLabelW: 90,     // 行ラベル (例: "T0 これ") の幅
  colLabelH: 18,     // 列ラベル (例: "d0") の高さ
  matrixGap: 30,     // 行列間の縦余白
  matrixTitleH: 18,  // 行列名 (例: "X") の高さ
  matrixShapeH: 14,  // 行列形状 (例: "(T × d_model)") の高さ
};

const SVG_NS = "http://www.w3.org/2000/svg";

function el(tag, attrs = {}, children = []) {
  const e = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  for (const c of (Array.isArray(children) ? children : [children])) {
    if (c == null) continue;
    if (typeof c === "string") e.appendChild(document.createTextNode(c));
    else e.appendChild(c);
  }
  return e;
}

/**
 * View のインスタンスを作成。SVG ルート要素と net (model.js) を受け取り、
 * pan/zoom 状態と現在表示中タブを内部に持つ。
 *
 * @param {SVGElement} svg
 * @param {object} net — model.js の createTransformer 戻り値
 * @param {object} [handlers] — { onCellClick({matrix,row,col}) }
 * @returns {object} view
 */
export function createView(svg, net, handlers = {}) {
  const view = {
    svg,
    net,
    activeTab: "embed",       // 'embed' | 'qkv' | 'attn' | 'out'
    viewBox: { x: 0, y: 0, w: 1000, h: 600 },
    naturalSize: { w: 1000, h: 600 },
    handlers,
    selection: null,          // { matrix, row, col }
  };
  installPanZoom(view);
  return view;
}

/**
 * 現在の選択 (= 「式の展開」で表示中のセル) を更新して再描画。
 */
export function setSelection(view, sel) {
  view.selection = sel;
  renderAll(view);
}

/**
 * net の現在の値で SVG を全描画。Forward 後・タブ切り替え後・Reset 後に呼ぶ。
 */
export function renderAll(view) {
  const { svg, net } = view;
  // 既存 children を全クリア
  while (svg.firstChild) svg.removeChild(svg.firstChild);

  // ルート <g>。pan/zoom は viewBox で制御するので transform は使わない。
  const g = el("g", { class: "g-content" });
  svg.appendChild(g);

  // タブごとに「描画する行列リスト」を組み立て、縦に並べて描画する。
  const matrices = matricesForTab(view.activeTab, net);
  let yCursor = 8;
  let maxRight = 0;
  for (const spec of matrices) {
    const drawn = drawMatrix(g, spec, yCursor, view);
    yCursor = drawn.bottomY + LAYOUT.matrixGap;
    maxRight = Math.max(maxRight, drawn.rightX);
  }

  // viewBox の自然サイズを計算してリセット
  view.naturalSize.w = Math.max(maxRight + 16, 600);
  view.naturalSize.h = Math.max(yCursor + 16, 400);
  // 現在の viewBox サイズを保ったまま svg にセットし直す
  // (初回 / Fit ボタン押下時は applyZoomFit() で自然サイズ全体を表示)
  applyViewBox(view);
}

/**
 * 表示するタブを切り替えて再描画。
 *
 * 注意: applyZoomFit は naturalSize を参照するので、必ず renderAll で
 * 新しい naturalSize を計算した後に呼ぶ。順序を逆にすると、古いタブの
 * naturalSize で viewBox を設定してしまい初期表示が歪む。
 */
export function setTab(view, tabName) {
  if (!["embed", "qkv", "attn", "out", "ffn"].includes(tabName)) {
    throw new Error(`Unknown tab: ${tabName}`);
  }
  view.activeTab = tabName;
  renderAll(view);     // まず新タブを描画 → naturalSize 更新
  applyZoomFit(view);  // 新しい naturalSize で全体表示にリセット
}

/**
 * pan/zoom: viewBox を初期 (= 全体表示) に戻す。
 */
export function applyZoomFit(view) {
  view.viewBox.x = 0;
  view.viewBox.y = 0;
  view.viewBox.w = view.naturalSize.w;
  view.viewBox.h = view.naturalSize.h;
  applyViewBox(view);
}

/**
 * pan/zoom: ズーム倍率を 1.0× に固定 (中心を保ったまま)。
 */
export function applyZoom100(view) {
  // viewBox の幅 = SVG の DOM 上の幅 にすれば 1px = 1 unit。
  const rect = view.svg.getBoundingClientRect();
  const cx = view.viewBox.x + view.viewBox.w / 2;
  const cy = view.viewBox.y + view.viewBox.h / 2;
  view.viewBox.w = rect.width || view.naturalSize.w;
  view.viewBox.h = rect.height || view.naturalSize.h;
  view.viewBox.x = cx - view.viewBox.w / 2;
  view.viewBox.y = cy - view.viewBox.h / 2;
  applyViewBox(view);
}

/**
 * 現在のズーム倍率を返す。
 * preserveAspectRatio="xMidYMid meet" は viewBox 全体を DOM 矩形に収めるよう
 * 一様縮小するので、実際のスケールは min(rect.w / vbW, rect.h / vbH)。
 * (X 軸比だけを返すと Y 軸が支配的なときに視覚スケールと大きくズレる。)
 */
export function currentZoom(view) {
  const rect = view.svg.getBoundingClientRect();
  if (!rect.width || !rect.height || !view.viewBox.w || !view.viewBox.h) return 1.0;
  const sx = rect.width  / view.viewBox.w;
  const sy = rect.height / view.viewBox.h;
  return Math.min(sx, sy);
}

/**
 * 中心アンカーで viewBox を factor 倍する (factor < 1 でズームイン)。
 * ホイールズームと同じロジックをマウス位置の代わりに「中央」固定で適用。
 */
export function applyZoomDelta(view, factor) {
  const cx = view.viewBox.x + view.viewBox.w / 2;
  const cy = view.viewBox.y + view.viewBox.h / 2;
  view.viewBox.w *= factor;
  view.viewBox.h *= factor;
  view.viewBox.x = cx - view.viewBox.w / 2;
  view.viewBox.y = cy - view.viewBox.h / 2;
  applyViewBox(view);
}

// ─── 内部ヘルパ ────────────────────────────────────────────

function applyViewBox(view) {
  const { x, y, w, h } = view.viewBox;
  view.svg.setAttribute("viewBox", `${x} ${y} ${w} ${h}`);
  view.svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
}

/**
 * 指定タブで表示する行列スペック群を組み立てる。
 * spec: { title, shape, rows, cols, data, mode, rowLabels, colLabels }
 *   mode = "signed" (青=負/赤=正) | "attn" (青濃淡 0..1)
 */
function matricesForTab(tab, net) {
  const T = net.T, D = net.d_model, K = net.d_k;
  const wordOf = (i) => net.vocab[net.tokens[i]] ?? "?";
  const tokenLabels = Array.from({ length: T }, (_, i) => `T${i} ${wordOf(i)}`);
  const dimLabels = (n) => Array.from({ length: n }, (_, i) => `d${i}`);

  if (tab === "embed") {
    // W_E から現在の tokens 分の行だけ抜き出した「ビュー」を一時的に作る。
    // Lesson 2 で「token 埋め込み」と「位置エンコーディング」を独立に観察するため、
    // X (= W_E + W_P) と並べて 3 行列を表示する。
    const WE_selected = new Float64Array(T * D);
    for (let t = 0; t < T; t++) {
      const tid = net.tokens[t];
      for (let d = 0; d < D; d++) {
        WE_selected[t * D + d] = net.W_E[tid * D + d];
      }
    }
    const posLabels = Array.from({ length: T }, (_, i) => `pos ${i}`);
    return [
      {
        matrixKey: "W_E",
        title: "W_E[tokens[t], :]   token 埋め込み (位置に依存せず token の意味だけ)",
        shape: `(T=${T}) × (d_model=${D})`,
        rows: T, cols: D, data: WE_selected, mode: "signed",
        rowLabels: tokenLabels, colLabels: dimLabels(D),
      },
      {
        matrixKey: "W_P",
        title: "W_P[t, :]   位置エンコーディング (token と無関係に位置だけで決まる sin/cos)",
        shape: `(T=${T}) × (d_model=${D})`,
        rows: T, cols: D, data: net.W_P, mode: "signed",
        rowLabels: posLabels, colLabels: dimLabels(D),
      },
      {
        matrixKey: "X",
        title: "X = W_E[tokens] + W_P   (上 2 つの和、Q/K/V 計算の入力)",
        shape: `(T=${T}) × (d_model=${D})`,
        rows: T, cols: D, data: net.X, mode: "signed",
        rowLabels: tokenLabels, colLabels: dimLabels(D),
      },
    ];
  }
  if (tab === "qkv") {
    // Tier 2 (Multi-Head): W_Q / W_K / W_V は global (d_model, h*d_k) で表示し、
    // 出力 Q / K / V は head 別に縦に並べる。各 head の cell は (T, d_k)。
    const block0 = net.blocks[0];
    const h = net.h;
    const HD = h * K;       // = d_model

    /** weight 行列の列ヘッダで「head 境界」を見せるためのラベル。
     * d_k=8 のとき: d0 d1 ... d7 | d0 d1 ... d7 を head 0/1 として表示。 */
    const colHeadLabels = (n) => Array.from({ length: n }, (_, c) => {
      const hi = Math.floor(c / K);
      const k  = c % K;
      return `h${hi}.d${k}`;
    });

    /** head hi の Q を (T, d_k) に切り出して spec を作る。 */
    const headSpec = (matrixKey, baseTitle, buf, hi) => ({
      matrixKey: `${matrixKey}_h${hi}`,
      title: `${baseTitle}   [head ${hi}]`,
      shape: `(T=${T}) × (d_k=${K})`,
      rows: T, cols: K,
      data: headSliceTHD(buf, T, h, K, hi),
      mode: "signed",
      rowLabels: tokenLabels, colLabels: dimLabels(K),
    });

    const specs = [
      { matrixKey: "W_Q",
        title: "W_Q   X[t] を Query 空間に射影する重み (preset で hand-crafted)",
        shape: `(d_model=${D}) × (h*d_k=${HD})`,
        rows: D, cols: HD, data: block0.W_Q, mode: "signed",
        rowLabels: dimLabels(D), colLabels: colHeadLabels(HD) },
    ];
    for (let hi = 0; hi < h; hi++) {
      specs.push(headSpec("Q", "Q = X · W_Q   各 token の「問い合わせ」", net.Q, hi));
    }
    specs.push({
      matrixKey: "W_K",
      title: "W_K   X[t] を Key 空間に射影する重み",
      shape: `(d_model=${D}) × (h*d_k=${HD})`,
      rows: D, cols: HD, data: block0.W_K, mode: "signed",
      rowLabels: dimLabels(D), colLabels: colHeadLabels(HD),
    });
    for (let hi = 0; hi < h; hi++) {
      specs.push(headSpec("K", "K = X · W_K   各 token の「ラベル」", net.K, hi));
    }
    specs.push({
      matrixKey: "W_V",
      title: "W_V   X[t] を Value 空間に射影する重み (Tier 2 では単位行列のまま)",
      shape: `(d_model=${D}) × (h*d_k=${HD})`,
      rows: D, cols: HD, data: block0.W_V, mode: "signed",
      rowLabels: dimLabels(D), colLabels: colHeadLabels(HD),
    });
    for (let hi = 0; hi < h; hi++) {
      specs.push(headSpec("V", "V = X · W_V   各 token が渡す「中身」", net.V, hi));
    }
    return specs;
  }
  if (tab === "attn") {
    // Tier 2: scores / attn は head 別に並べる。
    const h = net.h;
    const specs = [];
    for (let hi = 0; hi < h; hi++) {
      specs.push({
        matrixKey: `scores_h${hi}`,
        title: `scores = (Q · Kᵀ) / √d_k   [head ${hi}]`,
        shape: `(T=${T}) × (T=${T})`,
        rows: T, cols: T,
        data: headSliceHTT(net.scores, T, hi),
        mode: "signed",
        rowLabels: tokenLabels, colLabels: tokenLabels.slice(),
      });
      specs.push({
        matrixKey: `attn_h${hi}`,
        title: `attn = softmax(scores)   [head ${hi}]   (各行の和 = 1)`,
        shape: `(T=${T}) × (T=${T})`,
        rows: T, cols: T,
        data: headSliceHTT(net.attn, T, hi),
        mode: "attn",
        rowLabels: tokenLabels, colLabels: tokenLabels.slice(),
      });
    }
    return specs;
  }
  if (tab === "out") {
    // Tier 2: attnOut は head 別、Y は concat·W_O、residual1 は X+Y、ln1_out は LN(residual1)。
    const h = net.h;
    const specs = [];
    for (let hi = 0; hi < h; hi++) {
      specs.push({
        matrixKey: `attnOut_h${hi}`,
        title: `attnOut = attn · V   [head ${hi}]`,
        shape: `(T=${T}) × (d_k=${K})`,
        rows: T, cols: K,
        data: headSliceTHD(net.attnOut, T, h, K, hi),
        mode: "signed",
        rowLabels: tokenLabels, colLabels: dimLabels(K),
      });
    }
    specs.push({
      matrixKey: "Y",
      title: "Y = concat(attnOut) · W_O   (Multi-Head Attention の射影出力)",
      shape: `(T=${T}) × (d_model=${D})`,
      rows: T, cols: D, data: net.Y, mode: "signed",
      rowLabels: tokenLabels, colLabels: dimLabels(D),
    });
    specs.push({
      matrixKey: "residual1",
      title: "residual1 = X + Y   (残差接続: 元 token 情報を足し戻す)",
      shape: `(T=${T}) × (d_model=${D})`,
      rows: T, cols: D, data: net.residual1, mode: "signed",
      rowLabels: tokenLabels, colLabels: dimLabels(D),
    });
    specs.push({
      matrixKey: "ln1_out",
      title: "ln1_out = LayerNorm(residual1, γ=1, β=0)   (Attention 部の出力、各行の mean=0 / std≈1)",
      shape: `(T=${T}) × (d_model=${D})`,
      rows: T, cols: D, data: net.ln1_out, mode: "signed",
      rowLabels: tokenLabels, colLabels: dimLabels(D),
    });
    return specs;
  }
  if (tab === "ffn") {
    // FFN タブ: 重み (W1, W2) + 中間活性 (ffn_h) + 出力 (ffn_out) + 残差/LN2 を縦に並べる。
    const block0 = net.blocks[0];
    const d_ff = net.d_ff;
    const ffLabels = (n) => Array.from({ length: n }, (_, i) => `h${i}`);
    return [
      { matrixKey: "FFN_W1",
        title: "W1   ln1_out を d_ff 次元の検出器空間に射影する重み (preset で hand-crafted)",
        shape: `(d_model=${D}) × (d_ff=${d_ff})`,
        rows: D, cols: d_ff, data: block0.FFN_W1, mode: "signed",
        rowLabels: dimLabels(D), colLabels: ffLabels(d_ff) },
      { matrixKey: "ffn_h",
        title: "ffn_h = GELU(ln1_out · W1 + b1)   各 token の検出器ニューロン活性",
        shape: `(T=${T}) × (d_ff=${d_ff})`,
        rows: T, cols: d_ff, data: net.ffn_h, mode: "signed",
        rowLabels: tokenLabels, colLabels: ffLabels(d_ff) },
      { matrixKey: "FFN_W2",
        title: "W2   検出器活性を d_model に書き戻す重み",
        shape: `(d_ff=${d_ff}) × (d_model=${D})`,
        rows: d_ff, cols: D, data: block0.FFN_W2, mode: "signed",
        rowLabels: ffLabels(d_ff), colLabels: dimLabels(D) },
      { matrixKey: "ffn_out",
        title: "ffn_out = ffn_h · W2 + b2   FFN の出力",
        shape: `(T=${T}) × (d_model=${D})`,
        rows: T, cols: D, data: net.ffn_out, mode: "signed",
        rowLabels: tokenLabels, colLabels: dimLabels(D) },
      { matrixKey: "residual2",
        title: "residual2 = ln1_out + ffn_out   2 回目の残差接続",
        shape: `(T=${T}) × (d_model=${D})`,
        rows: T, cols: D, data: net.residual2, mode: "signed",
        rowLabels: tokenLabels, colLabels: dimLabels(D) },
      { matrixKey: "ln2_out",
        title: "ln2_out = LayerNorm(residual2, γ=1, β=0)   block 最終出力",
        shape: `(T=${T}) × (d_model=${D})`,
        rows: T, cols: D, data: net.ln2_out, mode: "signed",
        rowLabels: tokenLabels, colLabels: dimLabels(D) },
    ];
  }
  return [];
}

/**
 * 1 つの行列を SVG に描画。yTop から開始し、下端 / 右端の y, x を返す。
 * view: clickハンドラ呼び出しと selection ハイライトに使う。
 */
function drawMatrix(parent, spec, yTop, view) {
  const { matrixKey, title, shape, rows, cols, data, mode, rowLabels, colLabels } = spec;
  const sel = view?.selection;
  const isSelectedHere = sel && sel.matrix === matrixKey;

  // タイトル
  parent.appendChild(el(
    "text",
    { class: "matrix-label", x: 8, y: yTop + LAYOUT.matrixTitleH - 4 },
    title,
  ));
  parent.appendChild(el(
    "text",
    {
      class: "matrix-shape-label",
      x: 8,
      y: yTop + LAYOUT.matrixTitleH + LAYOUT.matrixShapeH - 4,
    },
    shape,
  ));

  // グリッドの起点
  const gridLeft = 8 + LAYOUT.rowLabelW;
  const gridTop = yTop + LAYOUT.matrixTitleH + LAYOUT.matrixShapeH + LAYOUT.colLabelH;

  // 値の絶対値の最大 (signed モードの色スケール用)
  let absMax = 0;
  for (let i = 0; i < rows * cols; i++) {
    const v = Math.abs(data[i]);
    if (v > absMax) absMax = v;
  }
  if (absMax === 0) absMax = 1;  // 全 0 のとき 0 除算回避

  // 列ヘッダ
  for (let c = 0; c < cols; c++) {
    parent.appendChild(el(
      "text",
      {
        class: "col-label",
        x: gridLeft + c * LAYOUT.cellW + LAYOUT.cellW / 2,
        y: gridTop - 4,
      },
      colLabels[c],
    ));
  }

  // 各行
  for (let r = 0; r < rows; r++) {
    const cellY = gridTop + r * LAYOUT.cellH;
    // 行ラベル
    parent.appendChild(el(
      "text",
      {
        class: "row-label",
        x: gridLeft - 6,
        y: cellY + LAYOUT.cellH / 2 + 4,
      },
      rowLabels[r],
    ));
    // セル
    for (let c = 0; c < cols; c++) {
      const cellX = gridLeft + c * LAYOUT.cellW;
      const v = data[r * cols + c];
      const fill = mode === "attn"
        ? attnFill(v)
        : signedFill(v, absMax);
      const isSelected = isSelectedHere && sel.row === r && sel.col === c;
      const cellClass = "cell-rect" + (isSelected ? " cell-selected" : "");
      const rect = el("rect", {
        class: cellClass,
        x: cellX, y: cellY,
        width: LAYOUT.cellW, height: LAYOUT.cellH,
        fill,
        "data-row": r, "data-col": c, "data-matrix": matrixKey,
      });
      // セルクリック → 選択 + handlers.onCellClick
      if (view && matrixKey) {
        rect.addEventListener("click", (ev) => {
          ev.stopPropagation();
          view.handlers?.onCellClick?.({ matrix: matrixKey, row: r, col: c });
        });
        // ドラッグ pan を妨げないよう、ホバー時はカーソル変更だけ。
        rect.style.cursor = "pointer";
      }
      parent.appendChild(rect);
      // セル内の数値
      const textClass = mode === "attn"
        ? (v >= 0.5 ? "cell-text attn-cell-text dark" : "cell-text attn-cell-text")
        : "cell-text";
      const txt = el("text", {
        class: textClass,
        x: cellX + LAYOUT.cellW / 2,
        y: cellY + LAYOUT.cellH / 2 + 1,
      }, formatCellValue(v, mode));
      txt.style.pointerEvents = "none";
      parent.appendChild(txt);
    }
  }

  const bottomY = gridTop + rows * LAYOUT.cellH;
  const rightX = gridLeft + cols * LAYOUT.cellW;
  return { bottomY, rightX };
}

function formatCellValue(v, mode) {
  if (!Number.isFinite(v)) return String(v);
  if (mode === "attn") return v.toFixed(2);
  // signed: 符号付き 3 桁
  const s = v.toFixed(3);
  return v >= 0 ? `+${s}` : s;
}

/**
 * signed モードのセル色 (青=負、白=0、赤=正、明度 = |v| / absMax)。
 */
function signedFill(v, absMax) {
  const t = Math.min(1, Math.abs(v) / absMax);  // 0..1
  // 透明度で表現するのは縞模様になるので、白から色へ補間
  if (v >= 0) {
    // 白 → 赤 (#c53030)
    const r = Math.round(255 - (255 - 197) * t);
    const g = Math.round(255 - (255 - 48) * t);
    const b = Math.round(255 - (255 - 48) * t);
    return `rgb(${r},${g},${b})`;
  } else {
    // 白 → 青 (#2b6cb0)
    const r = Math.round(255 - (255 - 43) * t);
    const g = Math.round(255 - (255 - 108) * t);
    const b = Math.round(255 - (255 - 176) * t);
    return `rgb(${r},${g},${b})`;
  }
}

/**
 * attn モードのセル色 (白=0、青濃=1)。0..1 の値を期待。
 */
function attnFill(v) {
  const t = Math.max(0, Math.min(1, v));
  const r = Math.round(255 - (255 - 43) * t);
  const g = Math.round(255 - (255 - 108) * t);
  const b = Math.round(255 - (255 - 176) * t);
  return `rgb(${r},${g},${b})`;
}

// ─── pan / zoom ────────────────────────────────────────────

function installPanZoom(view) {
  const svg = view.svg;
  let dragging = false;
  let lastClientX = 0, lastClientY = 0;

  svg.addEventListener("mousedown", (ev) => {
    if (ev.button !== 0) return;
    dragging = true;
    lastClientX = ev.clientX;
    lastClientY = ev.clientY;
    svg.classList.add("dragging");
  });

  window.addEventListener("mouseup", () => {
    if (dragging) {
      dragging = false;
      svg.classList.remove("dragging");
    }
  });

  window.addEventListener("mousemove", (ev) => {
    if (!dragging) return;
    const rect = svg.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    // 1 px の DOM 移動 = (viewBox.w / rect.width) ユニットの SVG 移動
    const dx = (ev.clientX - lastClientX) * (view.viewBox.w / rect.width);
    const dy = (ev.clientY - lastClientY) * (view.viewBox.h / rect.height);
    view.viewBox.x -= dx;
    view.viewBox.y -= dy;
    lastClientX = ev.clientX;
    lastClientY = ev.clientY;
    applyViewBox(view);
  });

  svg.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    const rect = svg.getBoundingClientRect();
    if (!rect.width || !rect.height) return;

    // ホイール方向: 正 = ズームアウト (= viewBox.w 拡大)、負 = ズームイン
    const factor = ev.deltaY > 0 ? 1.15 : 1 / 1.15;

    // マウス位置を中心にズーム
    const mx = (ev.clientX - rect.left) / rect.width;   // 0..1
    const my = (ev.clientY - rect.top) / rect.height;   // 0..1
    const targetX = view.viewBox.x + mx * view.viewBox.w;
    const targetY = view.viewBox.y + my * view.viewBox.h;

    const newW = view.viewBox.w * factor;
    const newH = view.viewBox.h * factor;
    view.viewBox.x = targetX - mx * newW;
    view.viewBox.y = targetY - my * newH;
    view.viewBox.w = newW;
    view.viewBox.h = newH;
    applyViewBox(view);
  }, { passive: false });
}
