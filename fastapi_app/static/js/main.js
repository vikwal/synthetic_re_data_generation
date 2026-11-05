// App shell: scenario sidebar, view routing, run orchestration.
import { api } from "./api.js?v=26";
import { h, toast, popMenu } from "./ui.js?v=26";
import { mountSetup } from "./setup.js?v=26";
import { mountResults, exportMenuItems } from "./results.js?v=26";
import { mountCompare } from "./compare.js?v=26";
import { mountModels } from "./models.js?v=26";
import { mountFL } from "./fl.js?v=26";

const app = { boot: null, scenarios: [], current: null, view: "workspace", mode: "setup", cleanup: null, running: false };
const $ = (id) => document.getElementById(id);
const main = $("main");

function teardown() { if (app.cleanup) { app.cleanup(); app.cleanup = null; } }
// Setup mode: the map fills the screen and both panels float above it.
function syncImmersive() {
  const on = app.view === "workspace" && app.mode === "setup" && !!app.current;
  const appEl = document.getElementById("app");
  appEl.classList.toggle("immersive", on);
  // Compare, Pre-trained model and FL model don't use the scenario list.
  appEl.classList.toggle("no-sidebar", app.view === "compare" || app.view === "models" || app.view === "fl");
}

async function refreshList() {
  app.scenarios = (await api.scenarios()).scenarios;
  renderList();
}

function renderList() {
  const ul = $("scenario-list");
  ul.replaceChildren(...app.scenarios.map((s) => {
    const li = h("li", { class: `scenario-item${app.current && app.current.id === s.id ? " active" : ""}`, onclick: () => openScenario(s.id) },
      h("span", { class: "ico" }, s.kind === "wind" ? "🌬️" : "☀️"),
      h("div", { class: "meta" }, h("div", { class: "name" }, s.name),
        h("div", { class: "sub" }, s.last_run_at ? `Run ${new Date(s.last_run_at).toLocaleDateString()}` : "Not run yet")),
      h("button", { class: "menu", title: "More", onclick: (e) => {
        e.stopPropagation();
        popMenu(e.currentTarget, [
          { label: "Rename", run: () => rename(s) },
          { label: "Duplicate", run: async () => { const d = await api.duplicateScenario(s.id); await refreshList(); openScenario(d.id); } },
          { label: "Delete", danger: true, run: () => remove(s) },
        ]);
      } }, "⋯"));
    return li;
  }));
  if (!app.scenarios.length) ul.append(h("li", { class: "hint", style: "padding:12px" }, "No scenarios yet. Create one with + Wind or + Solar."));
}

async function rename(s) {
  const name = prompt("Scenario name", s.name);
  if (!name || !name.trim()) return;
  await api.updateScenario(s.id, { name: name.trim() });
  if (app.current && app.current.id === s.id) app.current.name = name.trim();
  await refreshList();
  if (app.current && app.current.id === s.id) renderWorkspace();
}

async function remove(s) {
  if (!confirm(`Delete "${s.name}"? Its settings and all generated results will be permanently deleted.`)) return;
  await api.deleteScenario(s.id);
  if (app.current && app.current.id === s.id) { app.current = null; teardown(); }
  await refreshList();
  renderWorkspace();
}

async function create(kind) {
  const n = app.scenarios.filter((s) => s.kind === kind).length + 1;
  const sc = await api.createScenario(`${kind === "wind" ? "Wind park" : "Solar plant"} ${n}`, kind);
  await refreshList();
  openScenario(sc.id, "setup");
}

async function openScenario(id, mode) {
  app.current = await api.getScenario(id);
  app.mode = mode || (app.current.last_result_id ? "results" : "setup");
  app.view = "workspace";
  syncNav();
  renderList();
  renderWorkspace();
}

function syncNav() {
  document.querySelectorAll(".nav-btn").forEach((b) => b.classList.toggle("active", b.dataset.view === app.view));
}

function emptyState() {
  main.replaceChildren(h("div", { class: "empty" }, h("h2", {}, "Build your first scenario"),
    h("p", {}, "Create a wind park or a solar plant, place it on the map, and generate a synthetic time series."),
    h("div", { class: "btn-group", style: "justify-content:center;margin-top:16px" },
      h("button", { class: "btn primary", onclick: () => create("wind") }, "New wind park"),
      h("button", { class: "btn", onclick: () => create("solar") }, "New solar plant"))));
}

function renderWorkspace() {
  teardown();
  syncImmersive();
  if (app.view !== "workspace") return;
  const sc = app.current;
  if (!sc) return emptyState();

  const seg = h("div", { class: "seg" },
    ["setup", "results"].map((m) => h("button", {
      class: app.mode === m ? "active" : "", disabled: m === "results" && !sc.last_result_id,
      onclick: () => { app.mode = m; renderWorkspace(); },
    }, m === "setup" ? "Setup" : "Results")));
  const progress = h("div", { class: "run-progress", hidden: !app.running },
    h("div", { class: "bar" }, h("div", { id: "run-bar", style: "width:0%" })), h("div", { class: "run-msg", id: "run-msg" }));
  const runBtn = h("button", { class: "btn primary", id: "run-btn", disabled: app.running, onclick: runCurrent }, app.running ? "Generating…" : "▶ Generate");
  // Three-column grid: progress (left) · Setup/Results toggle (centre) · actions (right).
  const actions = [runBtn];
  if (sc.last_result_id) actions.push(h("button", { class: "btn", onclick: (e) => popMenu(e.currentTarget, exportMenuItems(sc.last_result_id)) }, "Export ▾"));
  const items = [h("div", { class: "scn-left" }, progress), seg, h("div", { class: "scn-actions" }, actions)];
  const body = h("div", { style: "flex:1;display:flex;flex-direction:column;min-height:0;overflow:auto" });
  main.replaceChildren(h("div", { class: "scn-head" }, items), body);

  if (app.mode === "results" && sc.last_result_id) {
    const wrap = h("div", { class: "results" });
    body.append(wrap);
    mountResults(wrap, sc.last_result_id);
  } else {
    body.style.overflow = "hidden";
    const view = h("div", { style: "flex:1;display:flex;min-height:0" });
    body.append(view);
    const m = mountSetup(view, sc, app.boot, () => refreshList());
    app.cleanup = m.destroy;
  }
}

async function runCurrent() {
  const sc = app.current;
  if (!sc || app.running) return;
  app.running = true;
  renderWorkspace();
  try {
    // Make sure latest edits are persisted before the server reads the scenario.
    await api.updateScenario(sc.id, { params: sc.params });
    let job = await api.run(sc.id);
    const t0 = Date.now();
    while (job.status === "queued" || job.status === "running") {
      const bar = $("run-bar"), msg = $("run-msg");
      if (bar) bar.style.width = `${Math.round(job.progress * 100)}%`;
      if (msg) msg.textContent = `${job.message || ""} · ${Math.round((Date.now() - t0) / 1000)} s`;
      await new Promise((r) => setTimeout(r, 500));
      job = await api.job(job.id);
    }
    if (job.status === "error") throw new Error(job.error);
    app.current = await api.getScenario(sc.id);
    app.mode = "results";
    toast("Generation complete", "ok");
    refreshList();
  } catch (e) {
    toast(e.message, "error");
  } finally {
    app.running = false;
    if (app.view === "workspace" && app.current && app.current.id === sc.id) renderWorkspace();
  }
}

async function showModels() {
  teardown();
  app.view = "models";
  syncNav();
  syncImmersive();
  const wrap = h("div", { style: "flex:1;display:flex;flex-direction:column;overflow:auto" });
  main.replaceChildren(wrap);
  await mountModels(wrap);
}

async function showFL() {
  teardown();
  app.view = "fl";
  syncNav();
  syncImmersive();
  const wrap = h("div", { style: "flex:1;display:flex;flex-direction:column;overflow:auto" });
  main.replaceChildren(wrap);
  mountFL(wrap);
}

async function showCompare() {
  teardown();
  app.view = "compare";
  syncNav();
  syncImmersive();
  const wrap = h("div", { style: "flex:1;display:flex;flex-direction:column;overflow:auto" });
  main.replaceChildren(wrap);
  await mountCompare(wrap);
}

async function init() {
  document.querySelectorAll(".nav-btn").forEach((b) => b.addEventListener("click", () => {
    if (b.dataset.view === "compare") showCompare();
    else if (b.dataset.view === "models") showModels();
    else if (b.dataset.view === "fl") showFL();
    else { app.view = "workspace"; syncNav(); renderWorkspace(); }
  }));
  $("new-wind").addEventListener("click", () => create("wind"));
  $("new-solar").addEventListener("click", () => create("solar"));
  try {
    app.boot = await api.bootstrap();
    await refreshList();
  } catch (e) {
    toast(`Failed to start: ${e.message}`, "error");
    return;
  }
  if (app.scenarios.length) openScenario(app.scenarios[0].id);
  else renderWorkspace();
}

init();
