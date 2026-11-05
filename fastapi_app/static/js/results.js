// Results dashboard: KPI cards + tabbed charts, explorer and data table.
import { api } from "./api.js?v=26";
import { h, toast, fmt } from "./ui.js?v=26";

const PLOT_CONFIG = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"] };

async function drawFigure(rid, name, params, plotEl) {
  plotEl.replaceChildren(h("div", { class: "note" }, "Loading…"));
  try {
    const f = await api.figure(rid, name, params);
    plotEl.replaceChildren();
    await Plotly.newPlot(plotEl, f.data, f.layout, PLOT_CONFIG);
  } catch (e) {
    Plotly.purge(plotEl);
    plotEl.replaceChildren(h("div", { class: "note" }, e.message));
  }
}

function figureCard(rid, name, { tall = false, tools = null, params = () => ({}) } = {}) {
  const plot = h("div", { class: "plot" });
  const reload = () => drawFigure(rid, name, params(), plot);
  const card = h("div", { class: `card${tall ? " tall" : ""}` }, tools ? tools(reload) : null, plot);
  // Lazy-load when the card becomes visible (tab panes start hidden).
  const io = new IntersectionObserver((entries) => {
    if (entries.some((e) => e.isIntersecting)) { io.disconnect(); reload(); }
  });
  queueMicrotask(() => io.observe(card));
  return card;
}

function kpiCards(res) {
  const k = res.kpis || {};
  const items = [
    ["Capacity", k.capacity_mw, "MW", 2],
    ["Annual energy", k.annual_energy_mwh, "MWh/yr", 0],
    ["Capacity factor", k.capacity_factor_pct, "%", 1],
    ["Full-load hours", k.full_load_hours, "h/yr", 0],
    ["Peak power", k.peak_power_mw, "MW", 2],
    ["Zero output", k.zero_output_pct, "% of time", 1],
  ];
  if (res.kind === "wind") {
    items.push(["Mean wind speed", k.mean_wind_speed, "m/s", 2]);
    const s = res.summary || {};
    if (s.wake_applied) items.push(["Park wake loss", s.park_wake_loss_pct, "%", 2]);
  } else {
    items.push(["Specific yield", k.specific_yield_kwh_kwp, "kWh/kWp", 0]);
  }
  return h("div", { class: "kpis" }, items.map(([l, v, u, d]) =>
    h("div", { class: "kpi" }, h("div", { class: "l" }, l), h("div", { class: "v" }, fmt(v, d), h("span", { class: "u" }, u)))));
}

function sitesTable(res) {
  const sites = res.sites || [];
  if (!sites.length) return null;
  const cols = [["#", (s, i) => i + 1], ["Model", (s) => s.turbine], ["Lat", (s) => fmt(s.lat, 4)], ["Lon", (s) => fmt(s.lon, 4)],
    ["Rated MW", (s) => fmt(s.rated_mw, 2)], ["Energy MWh", (s) => fmt(s.energy_mwh, 0)], ["CF %", (s) => fmt(s.capacity_factor_pct, 1)],
    ["Mean wind m/s", (s) => fmt(s.mean_wind_speed, 2)], ["Wake loss %", (s) => fmt(s.wake_loss_pct, 2)], ["Ageing loss %", (s) => fmt(s.ageing_loss_pct, 2)]];
  return h("div", { class: "card", style: "margin-top:16px" },
    h("div", { class: "table-wrap" }, h("table", { class: "data" },
      h("thead", {}, h("tr", {}, cols.map(([c]) => h("th", {}, c)))),
      h("tbody", {}, sites.map((s, i) => h("tr", {}, cols.map(([, f]) => h("td", {}, f(s, i)))))))));
}

function overviewPane(res) {
  const rid = res.id;
  const cards = [kpiCards(res)];
  const grid = h("div", { class: "grid2" });
  if (res.kind === "wind" && res.has_performance) {
    grid.append(figureCard(rid, "waterfall"), figureCard(rid, "loss-per-turbine"));
    if (res.summary.ageing_applied) grid.append(figureCard(rid, "ageing-timeline"));
  }
  grid.append(figureCard(rid, "seasonality"), figureCard(rid, "duration"));
  cards.push(grid);
  const t = sitesTable(res);
  if (t) cards.push(t);
  return cards;
}

function resourcePane(res) {
  const grid = h("div", { class: "grid2" });
  if (res.kind === "wind") {
    if (res.has_rose) grid.append(figureCard(res.id, "windrose", { tall: true }));
    grid.append(figureCard(res.id, "speed-hist", { tall: true }));
  } else grid.append(figureCard(res.id, "irradiance", { tall: true }));
  return [grid];
}

function performancePane(res) {
  return [h("div", { class: "grid2" },
    figureCard(res.id, "heatmap", { tall: true }),
    figureCard(res.id, "multi-year"))];
}

function select(options, value, onChange) {
  const s = h("select", {}, options.map(([v, l]) => h("option", { value: v, selected: String(v) === String(value) }, l)));
  s.addEventListener("change", () => onChange(s.value));
  return s;
}

function explorerPane(res) {
  const st = { date: res.dates[Math.floor(res.dates.length / 2)], feature: res.features[0] || "", year: res.years[0], turbine: res.turbines[0], hour: 12, hpYear: res.years[0], hbYear: res.years[0] };
  const shift = (days, reload) => {
    const d = new Date(st.date + "T00:00:00Z"); d.setUTCDate(d.getUTCDate() + days);
    const s = d.toISOString().slice(0, 10);
    if (res.dates.includes(s)) { st.date = s; dateInput.value = s; reload(); }
  };
  const dateInput = h("input", { type: "date", value: st.date, min: res.dates[0], max: res.dates[res.dates.length - 1] });
  const daily = figureCard(res.id, "daily", {
    tall: true, params: () => ({ date: st.date, feature: st.feature }),
    tools: (reload) => {
      dateInput.addEventListener("change", () => { st.date = dateInput.value; reload(); });
      return h("div", { class: "card-tools" }, h("b", {}, "Day"),
        h("button", { class: "btn small", onclick: () => shift(-1, reload) }, "◀"), dateInput,
        h("button", { class: "btn small", onclick: () => shift(1, reload) }, "▶"),
        res.features.length ? select(res.features.map((f) => [f, res.feature_labels[f] || f]), st.feature, (v) => { st.feature = v; reload(); }) : null);
    },
  });
  const quarterly = figureCard(res.id, "quarterly", {
    params: () => ({ year: st.year }),
    tools: (reload) => h("div", { class: "card-tools" }, h("b", {}, "Year"),
      select(res.years.map((y) => [y, y]), st.year, (v) => { st.year = Number(v); reload(); })),
  });
  const hourly = figureCard(res.id, "hour-profile", {
    params: () => ({ year: st.hpYear, hour: st.hour }),
    tools: (reload) => h("div", { class: "card-tools" }, h("b", {}, "Hour of day"),
      select(Array.from({ length: 24 }, (_, i) => [i, `${String(i).padStart(2, "0")}:00`]), st.hour, (v) => { st.hour = Number(v); reload(); }),
      h("b", {}, "Year"),
      select(res.years.map((y) => [y, y]), st.hpYear, (v) => { st.hpYear = Number(v); reload(); })),
  });
  const hourBox = figureCard(res.id, "hour-box", {
    params: () => ({ year: st.hbYear }),
    tools: (reload) => h("div", { class: "card-tools" }, h("b", {}, "Year"),
      select(res.years.map((y) => [y, y]), st.hbYear, (v) => { st.hbYear = Number(v); reload(); })),
  });
  const out = [h("div", { class: "grid2" }, daily, quarterly), h("div", { class: "grid2", style: "margin-top:16px" }, hourly, hourBox)];
  if (res.kind === "wind" && res.turbines.length) {
    out.push(h("div", { class: "grid2", style: "margin-top:16px" }, figureCard(res.id, "turbine-curve", {
      params: () => ({ turbine: st.turbine }),
      tools: (reload) => h("div", { class: "card-tools" }, h("b", {}, "Turbine"),
        select([["__park__", "Whole park"], ...res.turbines.map((t) => [t, t])], st.turbine, (v) => { st.turbine = v; reload(); })),
    })));
  }
  return out;
}

function dataPane(res) {
  const wrap = h("div", { class: "table-wrap" });
  const pager = h("div", { class: "pager" });
  let page = 1;
  async function load() {
    try {
      const d = await api.table(res.id, page);
      wrap.replaceChildren(h("table", { class: "data" },
        h("thead", {}, h("tr", {}, d.columns.map((c) => h("th", {}, c)))),
        h("tbody", {}, d.rows.map((r) => h("tr", {}, r.map((v) => h("td", {}, v === null ? "" : v)))))));
      pager.replaceChildren(
        h("button", { class: "btn small", disabled: page <= 1, onclick: () => { page--; load(); } }, "← Prev"),
        h("span", {}, `Page ${d.page} of ${d.pages} · ${fmt(d.total, 0)} rows`),
        h("button", { class: "btn small", disabled: page >= d.pages, onclick: () => { page++; load(); } }, "Next →"));
    } catch (e) { toast(e.message, "error"); }
  }
  load();
  return [pager, wrap];
}

export function exportMenuItems(rid) {
  const dl = (path) => () => { window.location.href = `/api/results/${rid}/${path}`; };
  return [
    { label: "Download CSV", run: dl("csv") },
    { label: "Download Parquet", run: dl("parquet") },
    { label: "Download summary (JSON)", run: dl("summary.json") },
    { label: "Print / save as PDF", run: () => window.print() },
  ];
}

export async function mountResults(container, rid) {
  container.replaceChildren(h("div", { class: "note" }, "Loading results…"));
  let res;
  try { res = await api.result(rid); }
  catch (e) { container.replaceChildren(h("div", { class: "note" }, e.message)); return; }

  const panes = [["Overview", overviewPane], ["Resource", resourcePane], ["Performance", performancePane], ["Explorer", explorerPane], ["Data", dataPane]];
  const tabBar = h("div", { class: "tabs" });
  const body = h("div");
  const built = {};
  function show(name) {
    [...tabBar.children].forEach((b) => b.classList.toggle("active", b.dataset.name === name));
    [...body.children].forEach((p) => { p.hidden = p.dataset.name !== name; });
    if (!built[name]) {
      const fn = panes.find(([n]) => n === name)[1];
      const pane = body.querySelector(`[data-name="${name}"]`);
      pane.replaceChildren(...fn(res));
      built[name] = true;
    }
    window.dispatchEvent(new Event("resize"));
  }
  panes.forEach(([n]) => {
    tabBar.append(h("button", { class: "tab", "data-name": n, onclick: () => show(n) }, n));
    body.append(h("div", { class: "tab-pane", "data-name": n, hidden: true }));
  });
  const stamp = res.created ? new Date(res.created).toLocaleString() : "";
  container.replaceChildren(
    h("p", { class: "hint" }, `${res.scenario_name} · generated ${stamp} · ${fmt(res.rows, 0)} rows`),
    tabBar, body);
  show("Overview");
}
