// Floating training-progress card. Lives on <body>, so it keeps running while you switch tabs
// (Workspace, Compare, Pre-trained model) and is restored after a page reload.
import { api } from "./api.js?v=26";
import { h } from "./ui.js?v=26";

const KEY = "synth_re_train_job";
let state = null;                       // { id, title, t0, job, minimised, done, error }
let timer = null, card = null;
let finishHandler = null;               // set by the Pre-trained model view; returns true if it showed the result itself
let pendingOpen = null;                 // model id to open next time the Models tab is mounted

export const setFinishHandler = (fn) => { finishHandler = fn; };
export const takePending = () => { const p = pendingOpen; pendingOpen = null; return p; };

function save() {
  try { state && !state.done && !state.error ? sessionStorage.setItem(KEY, JSON.stringify({ id: state.id, title: state.title, t0: state.t0 })) : sessionStorage.removeItem(KEY); } catch (_) { /* storage unavailable */ }
}

function dismiss() {
  clearInterval(timer); timer = null;
  if (card) card.remove();
  card = null; state = null; save();
}

function render() {
  if (!state) return;
  if (!card) { card = h("div", { class: "train-pop" }); document.body.append(card); }
  const { job, title, minimised, done, error } = state;
  const pct = done ? 100 : Math.round(((job && job.progress) || 0) * 100);
  const secs = Math.round((Date.now() - state.t0) / 1000);
  const elapsed = secs >= 60 ? `${Math.floor(secs / 60)} min ${secs % 60} s` : `${secs} s`;
  card.className = `train-pop${done ? " done" : ""}${error ? " failed" : ""}${minimised ? " mini" : ""}`;
  if (minimised) {
    card.replaceChildren(h("button", { class: "tp-mini", type: "button", title: "Show training progress", onclick: () => { state.minimised = false; render(); } },
      h("span", { class: "tp-ring" }), done ? "Model ready" : error ? "Training failed" : `Training ${pct}%`));
    return;
  }
  card.replaceChildren(...[
    h("div", { class: "tp-head" },
      h("b", {}, error ? "Training failed" : done ? "Model ready" : title),
      h("span", { class: "tp-actions" },
        done || error ? null : h("button", { type: "button", title: "Minimise", "aria-label": "Minimise", onclick: () => { state.minimised = true; render(); } }, "–"),
        done || error ? h("button", { type: "button", title: "Close", "aria-label": "Close", onclick: dismiss }, "×") : null)),
    error ? h("p", { class: "tp-msg err" }, error)
      : h("div", {},
        h("div", { class: "tp-bar" }, h("div", { style: `width:${pct}%` })),
        h("div", { class: "tp-row" }, h("span", { class: "tp-msg" }, done ? "Saved to your models." : (job && job.message) || "Starting…"),
          h("span", { class: "tp-pct" }, `${pct}%`)),
        done ? null : h("div", { class: "tp-sub" }, `Elapsed ${elapsed}`)),
    done ? h("div", { class: "tp-foot" }, h("button", { class: "btn small primary", type: "button", onclick: () => {
      pendingOpen = state.job.result_id;
      document.querySelector('.nav-btn[data-view="models"]')?.click();
      dismiss();
    } }, "View model")) : null,
  ].filter(Boolean));              // replaceChildren would print a null child as the text "null"
}

async function poll() {
  if (!state || state.done || state.error) return;
  try {
    const job = await api.job(state.id);
    state.job = job;
    if (job.status === "done") {
      state.done = true; save();
      const handled = finishHandler ? finishHandler(job) : false;
      if (handled) { dismiss(); return; }
    } else if (job.status === "error") {
      state.error = job.error || "Unknown error"; save();
    }
  } catch (_) { /* server busy: try again on the next tick */ }
  if (state && (state.done || state.error)) { clearInterval(timer); timer = null; }
  render();
}

export function trackJob(job, { title = "Training model" } = {}) {
  if (card) card.remove();
  card = null;
  state = { id: job.id, title, t0: Date.now(), job, minimised: false, done: false, error: null };
  save(); render();
  clearInterval(timer);
  timer = setInterval(poll, 800);
  poll();
}

// Resume after a page reload.
try {
  const saved = JSON.parse(sessionStorage.getItem(KEY) || "null");
  if (saved && saved.id) {
    state = { id: saved.id, title: saved.title, t0: saved.t0, job: null, minimised: false, done: false, error: null };
    render(); timer = setInterval(poll, 800); poll();
  }
} catch (_) { /* ignore */ }
