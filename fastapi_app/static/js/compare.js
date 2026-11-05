// Compare view with two modes: synthetic vs synthetic, and real vs synthetic.
import { api } from "./api.js?v=26";
import { h, toast, fmt } from "./ui.js?v=26";

const PLOT_CONFIG = { responsive: true, displaylogo: false };
const label = (r) => `${r.scenario_name} · ${r.kind} · ${new Date(r.created).toLocaleString()}`;

export function plotCards(figures) {
  const entries = Object.values(figures);
  const cards = entries.map(() => h("div", { class: "card" }, h("div", { class: "plot" })));
  requestAnimationFrame(() => entries.forEach((f, i) => Plotly.newPlot(cards[i].firstChild, f.data, f.layout, PLOT_CONFIG)));
  return h("div", { class: "grid2" }, cards);
}


export function helpButton(title, lines) {
  const pop = h("div", { class: "help-pop", hidden: true }, h("b", {}, title), h("ul", {}, lines.map((l) => h("li", {}, l))));
  const btn = h("button", { class: "help-btn", title: "Help", "aria-label": "Help", onclick: (e) => { e.stopPropagation(); pop.hidden = !pop.hidden; } }, "?");
  document.addEventListener("click", (e) => { if (!pop.hidden && !pop.contains(e.target)) pop.hidden = true; });
  return h("div", { class: "help-wrap" }, btn, pop);
}

export function filePicker(accept) {
  const input = h("input", { type: "file", accept, hidden: true });
  const name = h("span", { class: "file-name" }, "No file chosen");
  const box = h("button", { type: "button", class: "file-pick", onclick: () => input.click() },
    h("span", { class: "file-icon" }, "⇪"),
    h("span", { class: "file-text" }, name, h("span", { class: "file-sub" }, "Drop a file here or click to browse")),
    h("span", { class: "file-btn" }, "Choose file"));
  const show = () => {
    name.textContent = input.files.length ? input.files[0].name : "No file chosen";
    box.classList.toggle("has-file", input.files.length > 0);
  };
  input.addEventListener("change", show);
  // Drag and drop onto the box.
  box.addEventListener("dragover", (e) => { e.preventDefault(); box.classList.add("drag"); });
  box.addEventListener("dragleave", () => box.classList.remove("drag"));
  box.addEventListener("drop", (e) => {
    e.preventDefault(); box.classList.remove("drag");
    if (e.dataTransfer.files.length) { input.files = e.dataTransfer.files; show(); }
  });
  return { input, el: h("div", {}, box, input) };
}

// ---------------------------------------------------------------- synthetic vs synthetic
function syntheticMode(list, reload) {
  const counter = h("span", { class: "run-count" }, "0 of 6 selected");
  const compareBtn = h("button", { class: "btn primary", disabled: true, onclick: () => run() }, "Compare");
  const sync = () => {
    const n = checks.filter((c) => c.cb.checked).length;
    counter.textContent = `${n} of 6 selected`;
    compareBtn.disabled = n < 2 || n > 6;
    checks.forEach((c) => c.el.classList.toggle("selected", c.cb.checked));
  };
  const checks = list.map((r) => {
    const cb = h("input", { type: "checkbox", value: r.id });
    cb.addEventListener("change", sync);
    const el = h("label", { class: "run-tile" }, cb,
      h("div", { class: "run-info" },
        h("div", { class: "run-name" }, r.scenario_name),
        h("div", { class: "run-meta" }, h("span", { class: `kind-badge ${r.kind}` }, r.kind), new Date(r.created).toLocaleString())),
      h("button", { class: "run-del", title: "Delete this result", "aria-label": "Delete this result", onclick: async (e) => {
        e.preventDefault(); e.stopPropagation();
        if (!confirm(`Permanently delete the result "${r.scenario_name}" from ${new Date(r.created).toLocaleString()}?`)) return;
        try { await api.deleteResult(r.id); toast("Result deleted", "ok"); reload(); }
        catch (err) { toast(err.message, "error"); }
      } }, "✕"));
    return { cb, el };
  });
  const out = h("div", { class: "compare-out" });
  async function run() {
    const ids = checks.filter((c) => c.cb.checked).map((c) => c.cb.value);
    if (ids.length < 2) return toast("Select at least two results (max 6).", "error");
    if (ids.length > 6) return toast("At most 6 results can be compared.", "error");
    out.replaceChildren(h("div", { class: "note" }, "Comparing…"));
    try {
      const d = await api.compare(ids);
      const rows = [["Capacity (MW)", (i) => fmt(i.kpis.capacity_mw, 2)], ["Annual energy (MWh)", (i) => fmt(i.kpis.annual_energy_mwh, 0)],
        ["Capacity factor (%)", (i) => fmt(i.kpis.capacity_factor_pct, 1)], ["Full-load hours", (i) => fmt(i.kpis.full_load_hours, 0)],
        ["Peak power (MW)", (i) => fmt(i.kpis.peak_power_mw, 2)], ["Turbines", (i) => fmt(i.n_turbines, 0)],
        ["Wake loss (%)", (i) => fmt(i.wake_loss_pct, 2)], ["Ageing loss (%)", (i) => fmt(i.ageing_loss_pct, 2)]];
      const table = h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "Metric"), d.items.map((i) => h("th", {}, i.name)))),
        h("tbody", {}, rows.map(([l, f]) => h("tr", {}, h("td", {}, l), d.items.map((i) => h("td", {}, f(i)))))));
      out.replaceChildren(table, plotCards(d.figures));
    } catch (e) { out.replaceChildren(); toast(e.message, "error"); }
  }
  if (list.length < 2) {
    return h("div", {}, h("div", { class: "empty", style: "margin:30px auto" }, h("h2", {}, "Nothing to compare yet"),
      h("p", {}, "Run at least two scenarios (or the same one twice with different settings) and come back here.")),
      list.length ? h("div", { class: "run-grid" }, checks.map((c) => c.el)) : null);
  }
  return h("div", {},
    h("div", { class: "card run-picker" },
      h("div", { class: "run-picker-head" }, h("h3", {}, "Select runs to compare"), h("p", {}, "Pick two to six generated runs."), counter),
      h("div", { class: "run-grid" }, checks.map((c) => c.el)),
      h("div", { class: "run-picker-foot" }, compareBtn)),
    out);
}

// ---------------------------------------------------------------- real vs synthetic
function realMode(list) {
  if (!list.length) {
    return h("div", { class: "empty", style: "margin:30px auto" }, h("h2", {}, "Generate a synthetic run first"),
      h("p", {}, "Create and run a scenario for your plant's site, then upload your measured data here to compare."));
  }
  const pick = h("select", {}, list.map((r) => h("option", { value: r.id }, label(r))));
  const unit = { value: "W" };
  const unitSeg = h("div", { class: "seg unit-seg" }, ["W", "kW", "MW"].map((u) => h("button", {
    type: "button", class: u === unit.value ? "active" : "",
    onclick: (e) => { unit.value = u; [...e.currentTarget.parentNode.children].forEach((b) => b.classList.toggle("active", b === e.currentTarget)); },
  }, u)));
  const picker = filePicker(".csv,.txt,text/csv");
  const file = picker.input;
  const out = h("div", { class: "compare-out" });
  const btn = h("button", { class: "btn primary" }, "Compare");

  btn.addEventListener("click", async () => {
    if (!file.files.length) return toast("Choose your plant's CSV file first.", "error");
    btn.disabled = true;
    out.replaceChildren(h("div", { class: "note" }, "Comparing…"));
    try {
      const csv_text = await file.files[0].text();
      const d = await api.compareReal({ result_id: pick.value, csv_text, unit: unit.value });
      const m = d.metrics;
      const u = m.unit;
      const cards = [
        ["Energy difference", m.energy_diff_pct, "%", 1, "Total synthetic energy minus total real energy, as a share of the real energy. Positive: the synthetic data produces more."],
        ["Bias (synthetic − real)", m.bias, u, 3, "Average hourly power of the synthetic data minus the real data. Positive: synthetic is higher on average."],
        ["Bias", m.bias_pct, "%", 1, "The average bias as a percentage of the average real power."],
        ["MAE", m.mae, u, 3, "Mean absolute error: the typical size of the hourly difference, ignoring whether it is above or below."],
        ["RMSE", m.rmse, u, 3, "Root mean square error: like MAE, but large misses count more."],
        ["nRMSE", m.nrmse_pct, "% of capacity", 1, "RMSE as a share of the installed capacity, so parks of different size can be compared."],
        ["Correlation", m.correlation, "", 3, "How well the hourly shapes match, from -1 to 1. Close to 1: the ups and downs line up in time, even if the level differs."],
        ["Capacity factor (real)", m.cf_real_pct, "%", 1, "Real energy as a share of what the park would produce running at full capacity all the time."],
        ["Capacity factor (synthetic)", m.cf_synthetic_pct, "%", 1, "The same measure for the synthetic data."],
        ["Hours compared", m.hours, "h", 0, "Number of hours where both the real file and the synthetic run have data. Only these hours enter the metrics."],
      ];
      out.replaceChildren(
        h("div", { class: "overlap-chip", style: "display:inline-flex;align-items:center;gap:8px;margin:0 0 14px;padding:6px 12px;border-radius:999px;background:#ecfdf5;border:1px solid #a7f3d0;color:#065f46;font-size:13px;font-weight:600" },
          h("span", {}, `${fmt(m.hours, 0)} overlapping hours`),
          h("span", { style: "font-weight:400;color:#047857" }, `${m.start} to ${m.end} (UTC)`)),
        h("div", { class: "kpis" }, cards.map(([l, v, un, dg, tip]) =>
          h("div", { class: "kpi", title: tip, style: "cursor:help" }, h("div", { class: "l" }, l), h("div", { class: "v" }, fmt(v, dg), un ? h("span", { class: "u" }, un) : null)))),
        plotCards(d.figures));
    } catch (e) {
      out.replaceChildren();
      toast(e.message, "error");
    } finally { btn.disabled = false; }
  });

  return h("div", {},
    h("div", { class: "card run-picker real-form" },
      h("div", { class: "run-picker-head" }, h("h3", {}, "Compare with your measured data"),
        h("p", {}, "Choose the synthetic run for your site, then upload your plant's power file.")),
      h("div", { class: "real-cols" },
        h("div", { class: "real-col" },
          h("div", { class: "step-label" }, h("span", { class: "step-num" }, "1"), "Synthetic run"), pick,
          h("div", { class: "step-label", style: "margin-top:18px" }, h("span", { class: "step-num" }, "2"), "Power unit in your file"), unitSeg),
        h("div", { class: "real-col" },
          h("div", { class: "step-label" }, h("span", { class: "step-num" }, "3"), "Plant data file (CSV)"), picker.el)),
      h("div", { class: "run-picker-foot" }, btn)),
    out);
}

export async function mountCompare(container, mode = "synthetic") {
  container.replaceChildren(h("div", { class: "note" }, "Loading…"));
  let list;
  try { list = (await api.results()).results; } catch (e) { toast(e.message, "error"); return; }

  const seg = h("div", { class: "seg" },
    [["synthetic", "Synthetic vs synthetic"], ["real", "Real vs synthetic"]].map(([m, t]) =>
      h("button", { class: mode === m ? "active" : "", onclick: () => mountCompare(container, m) }, t)));
  const body = mode === "real" ? realMode(list) : syntheticMode(list, () => mountCompare(container, mode));
  const help = mode === "real"
    ? helpButton("Real vs synthetic", [
        "Upload measured data for the same site and period as a synthetic run.",
        "CSV with a timestamp column (UTC) and `power`, or `power_t1`, `power_t2`, … which are summed.",
        "`wind_speed` columns are optional. Semicolon or comma separated.",
        "Choose the power unit used in your file (W, kW or MW).",
        "Only hours where both datasets overlap are compared."])
    : helpButton("Synthetic vs synthetic", [
        "Tick two to six saved runs and press Compare.",
        "You get a KPI table and overlay charts for seasonality, monthly energy and duration curves.",
        "Use Delete to remove old results."]);
  container.replaceChildren(h("div", { class: "compare" },
    h("div", { class: "ml-head" }, h("div"), seg, h("div", { style: "justify-self:end" }, help)),
    body));
}
