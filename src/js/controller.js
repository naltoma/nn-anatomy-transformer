// controller.js — UI と model.js / view.js を接続するエントリポイント。
// P4 では最小限: DEFAULT_PRESET をロードして、Sample セレクタで選んだ文に対し
// forward を実行 → view.js で行列ヒートマップを描画する。
// インタラクション (セルクリック、式の展開) は P5 で追加。

import { DEFAULT_PRESET } from "./presets.js";
import { createTransformer, forward, reset } from "./model.js";
import { encode, SAMPLE_SENTENCES, VOCAB } from "./tokenizer.js";
import {
  createView,
  renderAll,
  setTab,
  setSelection,
  applyZoomFit,
  applyZoom100,
  applyZoomDelta,
  currentZoom,
} from "./view.js";
import { renderExplainHtml, renderAttentionMapHtml, escapeHtml, escapeAndBold } from "./explain.js";
import { LESSONS, getLesson } from "./lessons.js";

// ─── DOM ────────────────────────────────────────────────────

const $ = (id) => document.getElementById(id);

const ui = {
  lesson:       $("lesson"),
  sample:       $("sample"),
  btnForward:   $("btn-forward"),
  btnReset:     $("btn-reset"),
  phaseLabel:   $("phase-label"),

  svg:          $("net-svg"),
  btnZoomFit:   $("btn-zoom-fit"),
  btnZoom100:   $("btn-zoom-100"),
  btnZoomIn:    $("btn-zoom-in"),
  btnZoomOut:   $("btn-zoom-out"),
  zoomLabel:    $("zoom-label"),

  explainBody:  $("explain-body"),
  attmapBody:   $("attmap-body"),
  lessonBody:   $("lesson-body"),

  eventLog:     $("event-log"),
};

// ─── 状態 ───────────────────────────────────────────────────

const state = {
  net: null,
  view: null,
  sampleIdx: 0,
  selection: null,    // { matrix, row, col } — クリックされたセル
  currentLesson: null, // 選択中の Lesson (LESSONS のエントリ) | null
};

// ─── イベントログ ───────────────────────────────────────────

function log(msg) {
  const li = document.createElement("li");
  li.textContent = msg;
  ui.eventLog.appendChild(li);
  ui.eventLog.scrollTop = ui.eventLog.scrollHeight;
}

// ─── 初期化 ─────────────────────────────────────────────────

function init() {
  // Lesson セレクタ
  for (const lsn of LESSONS) {
    const opt = document.createElement("option");
    opt.value = lsn.id;
    opt.textContent = lsn.title;
    ui.lesson.appendChild(opt);
  }

  // Sample セレクタ
  for (let i = 0; i < SAMPLE_SENTENCES.length; i++) {
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = `#${i + 1}  ${SAMPLE_SENTENCES[i].join(" ")}`;
    ui.sample.appendChild(opt);
  }
  ui.sample.value = "0";

  // Net + view を構築
  state.net = createTransformer({ preset: DEFAULT_PRESET });
  state.view = createView(ui.svg, state.net, {
    onCellClick: onCellClick,
  });

  // 起動直後は最初のサンプルを net.tokens に書き込んでから (forward 前の状態で)
  // 描画する。これで Embed タブには W_E のままが見える状態になる。
  previewSampleInput();
  renderAll(state.view);
  // 初回ロード時は中央パネルを Fit 表示にする (viewBox は createView の初期値 1000×600 で
  // 立ち上がるが、Embed タブの自然サイズはそれと違うため)。
  // SVG のレイアウトが確定してからラベルを更新するため rAF を使う。
  applyZoomFit(state.view);
  requestAnimationFrame(updateZoomLabel);

  // タブ
  document.querySelectorAll(".view-tabs .tab").forEach((btn) => {
    btn.addEventListener("click", () => onViewTabClick(btn));
  });
  document.querySelectorAll(".side-tabs .tab").forEach((btn) => {
    btn.addEventListener("click", () => onSideTabClick(btn));
  });

  // ボタン
  ui.btnForward.addEventListener("click", onForward);
  ui.btnReset.addEventListener("click", onReset);
  ui.sample.addEventListener("change", onSampleChange);
  ui.lesson.addEventListener("change", onLessonChange);

  // pan/zoom コントロール
  ui.btnZoomFit.addEventListener("click", () => {
    applyZoomFit(state.view);
    updateZoomLabel();
  });
  ui.btnZoom100.addEventListener("click", () => {
    applyZoom100(state.view);
    updateZoomLabel();
  });
  // +10% は viewBox を 1/1.1 倍 (= 視覚を 1.1 倍に拡大)。−10% は逆。
  ui.btnZoomIn.addEventListener("click", () => {
    applyZoomDelta(state.view, 1 / 1.1);
    updateZoomLabel();
  });
  ui.btnZoomOut.addEventListener("click", () => {
    applyZoomDelta(state.view, 1.1);
    updateZoomLabel();
  });
  ui.svg.addEventListener("wheel", () => {
    // wheel 後のズーム倍率を表示更新 (view.js 側で viewBox は更新済み)
    requestAnimationFrame(updateZoomLabel);
  });
  // ドラッグ pan は viewBox.x/y のみ動かし倍率は変えないので更新不要

  // 補助パネルの「レッスン」タブに初期メッセージを描画
  renderLessonCard();

  log("[Init] preset=japanese-mini-v1, T=5, d_model=16, h=1");
}

function updateZoomLabel() {
  const z = currentZoom(state.view);
  ui.zoomLabel.textContent = `${z.toFixed(2)}×`;
}

// Sample 変更や Reset 後に「forward 前の状態」を準備する:
// - net.tokens に選択中サンプルの id 列を入れる
// - net.X 等は forward が走ってないのでクリア状態 (= reset() で 0 化)
function previewSampleInput() {
  const sample = SAMPLE_SENTENCES[state.sampleIdx];
  const ids = encode(sample);
  reset(state.net);
  // tokens は reset() で 0 にされるので、上書きする
  for (let t = 0; t < state.net.T; t++) state.net.tokens[t] = ids[t];
  // Embed タブに W_E + W_P が見えるよう、X だけ事前計算する。
  // (P4 では「Forward 前の状態でも何か絵が出る」のが大事。Q/K/V は 0 のまま)
  for (let t = 0; t < state.net.T; t++) {
    const off = t * state.net.d_model;
    const wOff = state.net.tokens[t] * state.net.d_model;
    const pOff = t * state.net.d_model;
    for (let d = 0; d < state.net.d_model; d++) {
      state.net.X[off + d] = state.net.W_E[wOff + d] + state.net.W_P[pOff + d];
    }
  }
  setPhase("idle (input previewed)");
}

function setPhase(label) {
  ui.phaseLabel.textContent = label;
}

function refreshExplain() {
  ui.explainBody.innerHTML = renderExplainHtml({
    net: state.net,
    selection: state.selection,
  });
}

function refreshAttentionMap() {
  ui.attmapBody.innerHTML = renderAttentionMapHtml(state.net);
}

function onCellClick({ matrix, row, col }) {
  state.selection = { matrix, row, col };
  setSelection(state.view, state.selection);
  refreshExplain();
  // クリックしたら自動で「式の展開」タブにフォーカスする
  activateSideTab("explain");
}

function activateSideTab(name) {
  document.querySelectorAll(".side-tabs .tab").forEach((b) => {
    b.classList.toggle("tab-active", b.getAttribute("data-side-tab") === name);
  });
  document.querySelectorAll("[data-side-panel]").forEach((p) => {
    p.classList.toggle("tab-panel-active", p.getAttribute("data-side-panel") === name);
  });
}

function activateViewTab(name) {
  document.querySelectorAll(".view-tabs .tab").forEach((b) => {
    b.classList.toggle("tab-active", b.getAttribute("data-view-tab") === name);
  });
  setTab(state.view, name);
  updateZoomLabel();
}

function renderLessonCard() {
  const lsn = state.currentLesson;
  if (!lsn) {
    ui.lessonBody.innerHTML = `
      <p class="muted">
        Lesson セレクタから Lesson 1〜Lesson 10 を選ぶと、ここに目的と「このレッスンでチェックすること (3 つ)」が表示されます。
      </p>`;
    return;
  }
  const checks = lsn.checks
    .map((c) => `<li>${escapeAndBold(c)}</li>`)
    .join("");
  ui.lessonBody.innerHTML = `
    <div class="lesson-card">
      <h3>${escapeHtml(lsn.title)}</h3>
      <p class="lesson-hint">${escapeAndBold(lsn.hint)}</p>
      <h4>このレッスンでチェックすること</h4>
      <ol class="lesson-checks">${checks}</ol>
    </div>`;
}

// (escapeHtml は explain.js から import 済み)

// ─── イベントハンドラ ────────────────────────────────────

function onSampleChange() {
  state.sampleIdx = parseInt(ui.sample.value, 10) || 0;
  state.selection = null;
  // Sample を変えても Lesson 選択は保持する。
  // (例: Lesson 2 Task 2 では「Sample を別文に切り替えても W_P が同じ」を観察する手順がある)
  // Lesson の sampleIdx と異なる Sample を選んだ場合は、現在の選択値で forward 状態を作り直す。
  previewSampleInput();
  if (state.currentLesson && state.currentLesson.runForward) {
    const ids = encode(SAMPLE_SENTENCES[state.sampleIdx]);
    forward(state.net, ids);
    setPhase("forward (sample change in lesson)");
  }
  renderAll(state.view);
  refreshExplain();
  refreshAttentionMap();
  log(`[Sample] #${state.sampleIdx + 1}: ${SAMPLE_SENTENCES[state.sampleIdx].join(" ")}`);
}

function onLessonChange() {
  const id = ui.lesson.value;
  if (!id) {
    state.currentLesson = null;
    renderLessonCard();
    return;
  }
  const lsn = getLesson(id);
  if (!lsn) return;
  state.currentLesson = lsn;
  // Lesson に従って Sample / View タブ / Forward を一括適用
  state.sampleIdx = lsn.sampleIdx;
  ui.sample.value = String(lsn.sampleIdx);
  state.selection = null;
  previewSampleInput();
  if (lsn.runForward) {
    const ids = encode(SAMPLE_SENTENCES[state.sampleIdx]);
    forward(state.net, ids);
    setPhase("forward (lesson auto)");
  } else {
    setPhase("idle (lesson preview)");
  }
  // 中央タブを Lesson 指定に合わせる
  activateViewTab(lsn.viewTab);
  // 補助パネルは「レッスン」タブをアクティブに
  activateSideTab("lesson");
  // 描画 / 各パネル更新
  renderAll(state.view);
  refreshExplain();
  refreshAttentionMap();
  renderLessonCard();
  log(`[Lesson] ${lsn.id}: ${lsn.title}`);
}

function onForward() {
  const sample = SAMPLE_SENTENCES[state.sampleIdx];
  const ids = encode(sample);
  forward(state.net, ids);
  setPhase("forward");
  renderAll(state.view);
  refreshExplain();
  refreshAttentionMap();
  // 主要 attention 関係をログに (Multi-Head: 各 head ごとに記録)
  const attn = state.net.attn;
  const T = state.net.T;
  const H = state.net.h;
  const headLabel = `[Forward] ${sample.join(" ")}`;
  for (let hi = 0; hi < H; hi++) {
    const sBase = hi * T * T;
    const lines = [];
    for (let i = 0; i < T; i++) {
      let bestJ = 0, bestV = -1;
      for (let j = 0; j < T; j++) {
        const v = attn[sBase + i * T + j];
        if (v > bestV) { bestV = v; bestJ = j; }
      }
      const queryWord = state.net.vocab[state.net.tokens[i]];
      const keyWord   = state.net.vocab[state.net.tokens[bestJ]];
      lines.push(`${queryWord} → ${keyWord} (${bestV.toFixed(2)})`);
    }
    log(`${headLabel} head${hi}: ${lines.join(",  ")}`);
  }
}

function onReset() {
  reset(state.net);
  state.selection = null;
  previewSampleInput();
  renderAll(state.view);
  refreshExplain();
  refreshAttentionMap();
  log(`[Reset]`);
}

function onViewTabClick(btn) {
  const tab = btn.getAttribute("data-view-tab");
  document.querySelectorAll(".view-tabs .tab").forEach((b) => {
    b.classList.toggle("tab-active", b === btn);
  });
  setTab(state.view, tab);
  updateZoomLabel();
}

function onSideTabClick(btn) {
  const tab = btn.getAttribute("data-side-tab");
  document.querySelectorAll(".side-tabs .tab").forEach((b) => {
    b.classList.toggle("tab-active", b === btn);
  });
  document.querySelectorAll("[data-side-panel]").forEach((p) => {
    p.classList.toggle("tab-panel-active", p.getAttribute("data-side-panel") === tab);
  });
}

// ─── 起動 ───────────────────────────────────────────────────

init();
