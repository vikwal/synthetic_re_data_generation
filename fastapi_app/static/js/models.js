// Pre-trained model tab: pre-train on generated data, then fine-tune on real measurements.
import { api } from "./api.js?v=26";
import { h, toast, fmt } from "./ui.js?v=26";
import { tr } from "./i18n.js?v=26";
import { plotCards, helpButton, filePicker } from "./compare.js?v=26";
import { openArchPopup } from "./archinfo.js?v=26";
import { trackJob, setFinishHandler, takePending } from "./trainpop.js?v=26";

const rl = (r) => `${r.scenario_name} · ${r.kind} · ${new Date(r.created).toLocaleString()}`;

function numField(label, value, opts = {}) {
  const input = h("input", { type: "number", value, min: opts.min, max: opts.max, step: opts.step ?? "any" });
  return { el: h("div", { class: "field" }, h("label", {}, label), input), get: () => Number(input.value), input };
}

// Hover card summarising a model: architecture, inputs, training data and settings.
function summaryPop(m, archs) {
  const info = archs.architectures.find((a) => a.id === m.architecture) || { label: m.architecture.toUpperCase(), description: "" };
  const hp = m.hyperparameters || m.hyperparams || {};
  const wxLabel = (id) => (archs.weather_sources || []).find((w) => w.id === id)?.label || (id === "era5" ? "Generated run (ERA5)" : id);
  const settings = m.architecture === "linear"
    ? `ridge α ${hp.ridge_alpha}`
    : `${hp.hidden} hidden · ${hp.layers} layers · dropout ${hp.dropout} · up to ${hp.epochs} epochs · lr ${hp.lr}`
      + (hp.early_stopping === false ? " · no early stopping" : ` · early stop after ${hp.patience || 6} epochs without gain`);
  const rows = [
    ["Task", `${m.kind} power ${((m.hyperparameters || m.hyperparams || {}).horizon || 1) > 1 ? `for ${(m.hyperparameters || m.hyperparams).horizon} hours at once` : "for one hour"} from the last ${m.lookback} h of weather${m.future_weather ? " and the weather forecast for those hours" : ""}` + (m.power_lag ? ` and the plant's own power up to ${m.power_lag} h earlier` : "")],
    ["Inputs", (m.features || []).map((x) => {                       // `<model>:<input>` (several models in one run) -> "input · Model"
      const i = x.indexOf(":");
      return i > 0 ? `${x.slice(i + 1)} · ${wxLabel(x.slice(0, i))}` : x;
    }).join(", ")],
    ["Weather model", (m.weather_sources && m.weather_sources.length ? m.weather_sources : [m.weather_source || "era5"]).map(wxLabel).join(" + ")
      + (m.weather_lead ? ` · forecasts issued ${m.weather_lead.from} to ${m.weather_lead.to} h ahead` : "")],
    [m.stage === "finetuned" ? "Weather from" : "Trained on", (m.source_names || []).filter(Boolean).join(", ") || "–"],
    ...(m.stage === "finetuned" ? [["Real data", m.real_file || "uploaded file"]] : []),
    ["Windows", `${fmt(m.n_train, 0)} train · ${fmt(m.n_val, 0)} validation`],
    ...(m.split && m.split.train && m.split.train.start
      ? [["Split", `Train ${m.split.train.start} to ${m.split.train.end} · validate ${m.split.val.start} to ${m.split.val.end}`]] : []),
    ["Settings", settings],
    ...(m.refit ? [["Final model", `Retrained on all data (train + validation) for ${m.refit.epochs} epochs, the best epoch found with early stopping. Metrics and plots are from the validation run.`]] : []),
  ];
  return h("div", { class: "model-pop" },
    h("div", { class: "pop-title" }, info.label, h("span", { class: "badge" }, m.stage === "finetuned" ? "Fine-tuned" : "Pre-trained")),
    h("div", { class: "pop-desc" }, info.description),
    h("dl", {}, rows.flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v)])));
}

let pendingResult = null;         // model to open right after the next mount
let activeTab = "train";          // survives re-mounts (e.g. after training finishes)

export async function mountModels(container) {
  container.replaceChildren(h("div", { class: "note" }, "Loading…"));
  let archs, results, models;
  try {
    [archs, results, models] = await Promise.all([api.mlArchitectures(), api.results(), api.mlModels()]);
  } catch (e) { toast(e.message, "error"); return; }
  results = results.results; models = models.models;
  const reload = () => mountModels(container);

  const help = helpButton("Pre-trained models", [
    "Pre-train: teach a model on generated data. Pick an architecture, tick one or more generated runs, set the lead time and the weather inputs (from one or more forecast models), then press Pre-train model.",
    "Models tab: your saved models. Hover a name for its details, press View for accuracy and plots.",
    "Fine-tune: adapt a saved model to your plant. Choose a generated run for the same site (it sets the location; the weather is the same forecast model(s) and lead time the model was pre-trained on), then upload your measured power file as a CSV.",
    "Not sure which architecture? Start with Linear regression for a quick baseline, then try TFT for the best accuracy.",
    "Lower error is better. Errors are in percent of your plant's rated capacity and are measured on the validation period (the days after the training end date up to the validation end date, or the last part of the data in By share mode), which the model never saw during training. After fine-tuning, the pre-trained-only errors show how much your own data improved the model.",
  ]);

  // ------------------------------------------------------------ 1. pre-train form
  // Pop-up with the layer diagram for the current settings. F = ticked weather inputs + 4 time columns + past power.
  // Multi-hour models also read the weather forecast for the predicted hours: a longer window and one extra flag input.
  const showArch = (a) => {
    const look = f.lookback.get() || 24, hor = f.horizon.get() || 1;
    openArchPopup(a, { L: look + (hor > 1 ? hor - 1 : 0), H: f.hidden.get() || 64, layers: f.layers.get() || 2, out: hor,
      F: featSel.size + 5 + (hor > 1 ? 1 : 0) });
  };
  let arch = (archs.architectures.find((a) => a.available) || archs.architectures[0]).id;
  // Architecture dropdown (unavailable ones are greyed out with the reason) plus the "How it works" pop-up.
  const archChip = h("span", { class: "ms-summary" });
  const archPanel = h("div", { class: "ms-panel", hidden: true });
  const archTrigger = h("button", { type: "button", class: "ms-trigger" }, archChip, h("span", { class: "ms-caret" }, "▾"));
  const archWrap = h("div", { class: "ms arch-ms" }, archTrigger, archPanel);
  const renderArch = () => {
    const cur = archs.architectures.find((a) => a.id === arch);
    archChip.replaceChildren(h("span", { class: "ms-chip" }, cur ? cur.label : "Select a model"));
    archPanel.replaceChildren(...archs.architectures.map((a) => h("button", { type: "button", class: `ms-option${a.id === arch ? " sel" : ""}`,
      disabled: !a.available, title: a.available ? "" : a.reason,
      onclick: () => { arch = a.id; archPanel.hidden = true; archTrigger.classList.remove("open"); renderArch(); syncHp(); } },
      h("span", { class: "feat-text" }, h("b", {}, a.label), a.available ? null : h("span", {}, a.reason)))));
  };
  archTrigger.addEventListener("click", () => { archPanel.hidden = !archPanel.hidden; archTrigger.classList.toggle("open", !archPanel.hidden); });
  document.addEventListener("click", (e) => { if (archWrap.isConnected && !archWrap.contains(e.target)) { archPanel.hidden = true; archTrigger.classList.remove("open"); } });
  renderArch();
  const archRow = h("div", { class: "arch-row" }, archWrap,
    h("button", { type: "button", class: "btn small", onclick: () => showArch(archs.architectures.find((a) => a.id === arch)) }, "How it works ›"));

  // Resource filter: only datasets of the chosen kind are listed (a model is trained on one kind).
  const counts = { wind: results.filter((r) => r.kind === "wind").length, solar: results.filter((r) => r.kind === "solar").length };
  let kind = counts.wind || !counts.solar ? "wind" : "solar";
  let picks = [];
  const dsBox = h("div", {});
  const kindSeg = h("div", { class: "seg", style: "align-self:flex-start" });

  // Input features the model may use. Everything is ticked by default; the list follows the chosen resource.
  const featCatalog = archs.features || { wind: [], solar: [], time: [] };
  const featBox = h("div", { class: "feat-box" });
  let featSel = new Set();
  let usePower = true;                                     // past power values as an input (on by default)
  let updateSplit = () => {};              // set further down once the split controls exist
  // Weather models (ICON-D2, ECMWF IFS, ...): all their inputs sit in one feature list that has a model filter.
  // Only real NWP forecasts are offered: the generated runs supply the power (target), not the weather.
  const sourcesList = (archs.weather_sources || []).filter((w) => w.id !== "era5" && !w.hidden);
  // Every height is used (100 m wind rather than 10 m when both exist), so there is nothing to choose.
  const defaultLayers = (avail) => avail.filter((l) => !(l === 10 && avail.includes(100)));
  // Feature ids are `<model>:<feature>`; the models used are the ones with at least one ticked input.
  const srcOf = (id) => id.slice(0, id.indexOf(":"));
  const usedSources = () => {
    const ids = new Set([...featSel].map(srcOf));
    const used = sourcesList.filter((s) => ids.has(s.id));
    return used.length ? used : sourcesList.filter((s) => s.available).slice(0, 1);
  };
  // Forecast lead time: how many hours before the predicted hour the forecast was issued.
  const leadFrom = numField("Lead time from (hours)", 12, { min: 0, max: 48, step: 1 });
  const leadTo = numField("Lead time to (hours)", 36, { min: 0, max: 48, step: 1 });
  let leadSource = null;
  const leadBox = h("div", { class: "lead-box" },
    h("div", { class: "lead-fields" }, leadFrom.el, leadTo.el));
  function syncLead() {
    const withLead = usedSources().filter((s) => s.package);
    leadBox.hidden = !withLead.length;
    if (withLead.length) {
      const cap = Math.min(...withLead.map((s) => s.lead.max));            // one window for all models: the shortest reach
      leadFrom.input.max = leadTo.input.max = cap;
      const key = withLead.map((s) => s.id).join("+");
      if (leadSource !== key) {
        leadFrom.input.value = withLead[0].lead.default[0]; leadTo.input.value = Math.min(withLead[0].lead.default[1], cap); leadSource = key;
      }
    }
  }
  function renderFeatures() {
    const key = `nwp_${kind}`;
    const ready = sourcesList.filter((s) => s.available);
    const multi = ready.length > 1;
    const base = [];
    for (const s of ready) {                                  // a source may lack some columns
      for (const f of featCatalog[key]) {
        if (s.features && !s.features.includes(f.id)) continue;
        base.push({ ...f, id: `${s.id}:${f.id}`, src: s.id, label: multi ? `${f.label} · ${s.label}` : f.label });
      }
    }
    const heights = [...new Set(base.map((f) => f.layer).filter((l) => l !== null && l !== undefined))].sort((x, y) => x - y);
    const layers = new Set(defaultLayers(heights));
    const list = base.filter((f) => f.layer == null || layers.has(f.layer));
    featSel = new Set(list.filter((f) => ready.length && f.src === ready[0].id).map((f) => f.id));   // default: the first model's inputs
    let filterSrc = null;                                      // model filter of the dropdown (null = all models)
    // Weather inputs as a multiselect dropdown (checkbox list inside a popover).
    const summary = h("span", { class: "ms-summary" });
    const trigger = h("button", { type: "button", class: "ms-trigger" }, summary, h("span", { class: "ms-caret" }, "▾"));
    const panel = h("div", { class: "ms-panel", hidden: true });
    const boxes = new Map(), rows = new Map();
    const shown = () => list.filter((f) => !filterSrc || f.src === filterSrc);
    const refresh = () => {                                    // every selected input shown as a green-bordered chip
      const chosen = list.filter((f) => featSel.has(f.id));
      summary.replaceChildren(...(chosen.length ? chosen.map((f) => h("span", { class: "ms-chip" }, f.label,
        h("button", { type: "button", class: "ms-x", title: `Remove ${f.label}`, "aria-label": `Remove ${f.label}`,
          onclick: (e) => { e.stopPropagation(); featSel.delete(f.id); const cb = boxes.get(f.id); if (cb) cb.checked = false; refresh(); } }, "×")))
        : [h("span", { class: "ms-empty" }, "Select weather inputs")]));
      syncLead(); updateSplit();
    };
    const setAll = (on) => {                                   // acts on the inputs currently shown by the filter
      shown().forEach((f) => { if (on) featSel.add(f.id); else featSel.delete(f.id); const cb = boxes.get(f.id); if (cb) cb.checked = on; });
      refresh();
    };
    // Model filter: buttons are toggled in place (not rebuilt), so a click never detaches its own target.
    const filterBtns = [[null, "All models"], ...ready.map((s) => [s.id, s.label])].map(([id, text]) => {
      const b = h("button", { type: "button", class: `layer-chip${id === null ? " on" : ""}`,
        onclick: () => {
          filterSrc = id;
          filterBtns.forEach(([bid, btn]) => btn.classList.toggle("on", bid === id));
          list.forEach((f) => { rows.get(f.id).style.display = !filterSrc || f.src === filterSrc ? "" : "none"; });
        } }, text);
      return [id, b];
    });
    // One toolbar row: model filter on the left, Select all / Clear on the right, all in the same small size.
    panel.append(h("div", { class: "ms-actions ms-bar" },
        h("div", { class: "ms-filter" }, ...(multi ? filterBtns.map(([, b]) => b) : [])),
        h("div", { class: "ms-acts" },
          h("button", { type: "button", class: "btn small", onclick: () => setAll(true) }, "Select all"),
          h("button", { type: "button", class: "btn small", onclick: () => setAll(false) }, "Clear"))),
      ...list.map((f) => {
        const cb = h("input", { type: "checkbox", checked: featSel.has(f.id) });
        boxes.set(f.id, cb);
        cb.addEventListener("change", () => { if (cb.checked) featSel.add(f.id); else featSel.delete(f.id); refresh(); });
        const row = h("label", { class: "ms-option" }, cb, h("span", { class: "feat-text" }, h("b", {}, f.label), h("span", {}, f.hint)));
        rows.set(f.id, row);
        return row;
      }));
    const wrap = h("div", { class: "ms" }, trigger, panel);
    trigger.addEventListener("click", () => { panel.hidden = !panel.hidden; trigger.classList.toggle("open", !panel.hidden); });
    document.addEventListener("click", (e) => { if (!wrap.isConnected) return; if (!wrap.contains(e.target)) { panel.hidden = true; trigger.classList.remove("open"); } });
    refresh();
    const group = () => h("div", { class: "feat-group" }, h("div", { class: "feat-title" }, "Weather"), wrap);
    // Recent power: past values of the plant's own output as an extra input (a classic time-series feature).
    const powerBox = h("input", { type: "checkbox", checked: usePower });
    powerBox.addEventListener("change", () => { usePower = powerBox.checked; });
    const powerChip = h("label", { class: "feat-chip power-chip" }, powerBox,
      h("span", { class: "feat-text" }, h("b", {}, "Past power values"),
        h("span", {}, "Use the plant's own power from the past hours as an input.")));
    featBox.replaceChildren(group(),
      h("div", { class: "feat-group" }, h("div", { class: "feat-title" }, "Power history"), powerChip));
  }

  function renderDatasets() {
    const list = results.filter((r) => r.kind === kind);
    // Generated runs of the chosen kind as a multiselect: ticked runs appear as green chips with an x.
    const summary = h("span", { class: "ms-summary" });
    const panel = h("div", { class: "ms-panel", hidden: true });
    const trigger = h("button", { type: "button", class: "ms-trigger" }, summary, h("span", { class: "ms-caret" }, "▾"));
    const wrap = h("div", { class: "ms" }, trigger, panel);
    const refresh = () => {
      const chosen = picks.filter((p) => p.cb.checked);
      summary.replaceChildren(...(chosen.length ? chosen.map(({ r, cb }) => h("span", { class: "ms-chip" }, r.scenario_name,
        h("button", { type: "button", class: "ms-x", title: `Remove ${r.scenario_name}`, "aria-label": `Remove ${r.scenario_name}`,
          onclick: (e) => { e.stopPropagation(); cb.checked = false; refresh(); updateSplit(); } }, "×")))
        : [h("span", { class: "ms-empty" }, list.length ? "Select generated data" : `No generated ${kind} data yet`)]));
    };
    picks = list.map((r) => {
      const cb = h("input", { type: "checkbox", value: r.id });
      cb.addEventListener("change", () => { refresh(); updateSplit(); });
      return { r, cb };
    });
    if (picks.length) picks[0].cb.checked = true;            // default: the most recent run of this kind
    panel.append(...(picks.length
      ? picks.map(({ r, cb }) => h("label", { class: "ms-option" }, cb,
        h("span", { class: "feat-text" }, h("b", {}, r.scenario_name),
          h("span", {}, `${new Date(r.created).toLocaleString()}${r.start ? ` · ${r.start} to ${r.end}` : ""}`))))
      : [h("span", { class: "hint", style: "margin:6px" }, `No generated ${kind} datasets yet. Run a ${kind} scenario in the Workspace first.`)]));
    trigger.addEventListener("click", () => { if (picks.length) { panel.hidden = !panel.hidden; trigger.classList.toggle("open", !panel.hidden); } });
    document.addEventListener("click", (e) => { if (!wrap.isConnected) return; if (!wrap.contains(e.target)) { panel.hidden = true; trigger.classList.remove("open"); } });
    dsBox.replaceChildren(wrap);
    refresh();
    kindSeg.replaceChildren(...["wind", "solar"].map((k) => h("button", { type: "button", class: kind === k ? "active" : "",
      onclick: () => { kind = k; renderDatasets(); } }, `${k === "wind" ? "Wind" : "Solar"} (${counts[k]})`)));
    renderFeatures();
    updateSplit();
  }
  renderDatasets();

  const nameIn = h("input", { placeholder: "e.g. GRU wind baseline" });
  const f = {
    lookback: numField("Lookback window (hours)", 24, { min: 1, max: 168, step: 1 }),
    epochs: numField("Epochs", 30, { min: 1, max: 300, step: 1 }),
    hidden: numField("Hidden size", 64, { min: 8, max: 256, step: 4 }),
    layers: numField("Layers", 2, { min: 1, max: 4, step: 1 }),
    dropout: numField("Dropout", 0.1, { min: 0, max: 0.6, step: 0.05 }),
    lr: numField("Learning rate", 0.001, { min: 0.000001, max: 1, step: 0.0001 }),
    batch: numField("Batch size", 128, { min: 16, max: 2048, step: 16 }),
    val: numField("Validation share", 0.2, { min: 0.05, max: 0.5, step: 0.05 }),
    horizon: numField("Forecast horizon (hours)", 24, { min: 1, max: 72, step: 1 }),
    alpha: numField("Ridge strength α", 1, { min: 0.0000001, step: 0.1 }),
  };
  const scalingSel = h("select", {}, [["standard", "Standardise"], ["minmax", "Min–max"], ["none", "None"]]
    .map(([v, l]) => h("option", { value: v }, l)));
  f.scaling = { el: h("div", { class: "field" }, h("label", {}, "Input scaling"), scalingSel), get: () => scalingSel.value };
  // Early stopping (neural networks only)
  const esCb = h("input", { type: "checkbox", checked: true });
  const toggle = (cb, text, title) => h("div", { class: "field" }, h("label", {}, "\u00a0"),
    h("label", { class: "toggle-box", title: title || "" }, cb, text));
  f.es = { el: toggle(esCb, "Early stopping", "Stop training when the monitored value stops improving."), get: () => esCb.checked };
  f.patience = numField("Patience (epochs)", 6, { min: 1, max: 50, step: 1 });
  const monitorSel = h("select", {}, [["val_rmse", "Validation RMSE"], ["val_mae", "Validation MAE"], ["train_loss", "Training RMSE"]]
    .map(([v, l]) => h("option", { value: v }, l)));
  f.monitor = { el: h("div", { class: "field" }, h("label", {}, "Monitor"), monitorSel), get: () => monitorSel.value };
  const refitCb = h("input", { type: "checkbox", checked: true });
  f.refit = { el: toggle(refitCb, "Retrain on all data", "After early stopping finds the best epoch, a fresh model is trained on train + validation data for exactly that many epochs."), get: () => refitCb.checked };
  f.delta = numField("Delta (min. improvement)", 0.001, { min: 0, max: 10, step: 0.001 });
  // Train / validation split: by date (default) or by share of the data. Drawn as a timeline bar.
  let splitMode = "date", dateTouched = false;
  const dateIn = h("input", { type: "date" }), fromIn = h("input", { type: "date" }), valIn = h("input", { type: "date" });
  let fromTouched = false, valTouched = false;
  const splitInfo = h("p", { class: "split-info" });
  const splitSeg = h("div", { class: "seg" });
  const trainSeg = h("div", { class: "split-train" }), valSeg = h("div", { class: "split-val" });
  const loLabel = h("span", {}), hiLabel = h("span", {});
  const dateWrap = h("div", { class: "split-fields" },
    h("label", { class: "sd-field" }, h("span", {}, "Train from"), fromIn),
    h("label", { class: "sd-field" }, h("span", {}, "Train until"), dateIn),
    h("label", { class: "sd-field" }, h("span", {}, "Validate until"), valIn));
  const splitBar = h("div", { class: "split-bar" }, trainSeg, valSeg);
  const splitDates = h("div", { class: "split-dates" }, loLabel, hiLabel);          // first and last day of the data, under the bar
  const iso = (d) => d.toISOString().slice(0, 10);
  const addDays = (iso0, n) => iso(new Date(new Date(iso0 + "T00:00:00Z").getTime() + n * 86400000));
  const dayDiff = (a, b) => Math.round((new Date(b) - new Date(a)) / 86400000);
  function selectedPeriod() {
    const runs = picks.filter((p) => p.cb.checked).map((p) => p.r).filter((r) => r.start && r.end);
    if (!runs.length) return null;
    let lo = runs.map((r) => r.start).sort()[0], hi = runs.map((r) => r.end).sort().slice(-1)[0];
    return lo < hi ? { lo, hi } : null;                       // the generated data sets the dates; weather gaps are skipped
  }
  const setBar = (trainPct, trainText, valText) => {
    trainSeg.style.width = `${trainPct}%`; valSeg.style.width = `${100 - trainPct}%`;
    trainSeg.textContent = trainText; valSeg.textContent = valText;
  };
  updateSplit = () => {
    splitSeg.replaceChildren(...[["date", "By date"], ["share", "By share"]].map(([m, t]) =>
      h("button", { type: "button", class: splitMode === m ? "active" : "", onclick: () => { splitMode = m; updateSplit(); } }, t)));
    const per = selectedPeriod();
    f.val.el.style.display = splitMode === "share" ? "" : "none";
    splitDates.style.display = dateWrap.style.display = splitMode === "date" ? "" : "none";
    loLabel.textContent = per ? per.lo : ""; hiLabel.textContent = per ? per.hi : "";
    if (splitMode === "share") {
      const v = Math.min(0.5, Math.max(0.05, f.val.get() || 0.2));
      setBar(100 * (1 - v), "Training", "Validation");
      splitInfo.textContent = `The last ${Math.round(v * 100)}% of the data (by time) is used for validation.`;
      return;
    }
    if (!per) {
      dateIn.disabled = true; fromIn.disabled = true; valIn.disabled = true;
      setBar(80, "Training", "Validation");
      splitInfo.textContent = picks.some((p) => p.cb.checked)
        ? "The selected data and weather model share no dates."
        : "Select a generated dataset above to choose the split date.";
      return;
    }
    dateIn.disabled = false; fromIn.disabled = false; valIn.disabled = false;
    fromIn.min = per.lo; fromIn.max = addDays(per.hi, -2);
    if (!fromTouched || !fromIn.value || fromIn.value < fromIn.min || fromIn.value > fromIn.max) fromIn.value = per.lo;
    dateIn.min = addDays(fromIn.value, 1); dateIn.max = addDays(per.hi, -1);
    if (!dateTouched || !dateIn.value || dateIn.value < dateIn.min || dateIn.value > dateIn.max) {
      dateIn.value = addDays(per.lo, Math.max(1, Math.round(dayDiff(per.lo, per.hi) * 0.8)));      // default: 80 % of the period
    }
    valIn.min = addDays(dateIn.value, 1); valIn.max = per.hi;                 // validation runs from the day after training to this date
    if (!valTouched || !valIn.value || valIn.value < valIn.min || valIn.value > valIn.max) valIn.value = per.hi;
    const nTrain = dayDiff(fromIn.value, dateIn.value) + 1, nVal = dayDiff(addDays(dateIn.value, 1), valIn.value) + 1;
    const skipped = dayDiff(per.lo, fromIn.value), skippedEnd = dayDiff(valIn.value, per.hi);
    setBar(100 * nTrain / (nTrain + nVal), "Training", "Validation");
    const wxNote = usedSources().filter((x) => x.start && (x.start > per.lo || x.end < per.hi))
      .map((x) => ` · ${x.label} only covers ${x.start} to ${x.end}; hours outside that are skipped`).join("");
    splitInfo.textContent = `Training ${fromIn.value} to ${dateIn.value} (${nTrain} days) · Validation ${addDays(dateIn.value, 1)} to ${valIn.value} (${nVal} days)` +
      (skipped ? ` · ${skipped} earlier days unused` : "") + (skippedEnd ? ` · ${skippedEnd} later days unused` : "") + wxNote;
  };
  dateIn.addEventListener("change", () => { dateTouched = true; updateSplit(); });
  fromIn.addEventListener("change", () => { fromTouched = true; updateSplit(); });
  valIn.addEventListener("change", () => { valTouched = true; updateSplit(); });
  f.val.input.addEventListener("input", () => updateSplit());
  const splitCard = h("div", { class: "split-card" }, splitSeg, splitBar, splitDates, dateWrap, splitInfo, f.val.el);
  // Settings in three labelled rows so every column lines up: input window, network, early stopping.
  const hpGrid = h("div", { class: "hp-groups" });
  const GROUPS = {
    nn: [["Input", ["lookback", "horizon", "scaling"]], ["Network", ["epochs", "hidden", "layers", "dropout", "lr", "batch"]],
      ["Early stopping", ["es", "patience", "monitor", "delta", "refit"]]],
    linear: [["Input", ["lookback", "horizon", "scaling"]], ["Model", ["alpha"]]],
  };
  function syncHp() {
    hpGrid.replaceChildren(...GROUPS[arch === "linear" ? "linear" : "nn"].map(([title, keys]) =>
      h("div", { class: "hp-group" }, h("div", { class: "feat-title" }, title),
        h("div", { class: "hp-row", style: `grid-template-columns:repeat(${Math.max(keys.length, 3)},minmax(0,1fr))` }, keys.map((k) => f[k].el)))));
  }
  syncHp();
  updateSplit();

  const trainBtn = h("button", { class: "btn primary" }, "Pre-train model");
  trainBtn.addEventListener("click", async () => {
    const ids = picks.filter((p) => p.cb.checked).map((p) => p.r.id);
    if (!ids.length) return toast("Choose at least one generated dataset.", "error");
    if (!featSel.size) return toast("Select at least one weather input.", "error");
    trainBtn.disabled = true;
    try {
      const used = usedSources();                                // models with at least one ticked input
      const body = { name: nameIn.value.trim() || null, architecture: arch, result_ids: ids, features: [...featSel].map((id) => (used.length === 1 ? id.slice(id.indexOf(":") + 1) : id)), weather_source: used[0].id, weather_sources: used.map((s) => s.id), lead_from: leadFrom.get(), lead_to: leadTo.get(),
        power_lag_hours: usePower ? 1 : 0, lookback: f.lookback.get(), val_frac: f.val.get(), train_from: splitMode === "date" && fromIn.value && !fromIn.disabled ? fromIn.value : null, train_until: splitMode === "date" && dateIn.value && !dateIn.disabled ? dateIn.value : null, val_until: splitMode === "date" && valIn.value && !valIn.disabled ? valIn.value : null, horizon: f.horizon.get(), scaling: f.scaling.get(),
        ...(arch === "linear" ? { ridge_alpha: f.alpha.get() }
          : { epochs: f.epochs.get(), hidden: f.hidden.get(), layers: f.layers.get(), dropout: f.dropout.get(), lr: f.lr.get(), batch_size: f.batch.get(),
            early_stopping: f.es.get(), patience: f.patience.get(), monitor: f.monitor.get(), min_delta: f.delta.get(), refit_full: f.refit.get() }) };
      const info = archs.architectures.find((a) => a.id === arch);
      trackJob(await api.mlTrain(body), { title: `Training ${info ? info.label : "model"}` });
      toast("Training started. Progress is shown in the corner.", "ok");
    } catch (e) { toast(e.message, "error"); } finally { trainBtn.disabled = false; }
  });

  // Step heading with an info icon: hovering (or focusing) the "i" explains what the setting means.
  function step(n, text, tip) {
    return h("div", { class: "step-label" }, h("span", { class: "step-num" }, n), text,
      tip ? h("span", { class: "info-tip", tabindex: "0", role: "img", "aria-label": tip, "data-tip": tip }, "i") : null);
  }
  const trainCard = h("div", { class: "card ml-card run-picker" },
    h("div", { class: "run-picker-head" }, h("h3", {}, "Pre-train on synthetic data"),
      h("p", {}, "Teach a model on generated runs, then fine-tune it on your own measurements.")),
    h("div", { class: "field" }, step(1, "Architecture", "The kind of model that is trained. Linear regression is a fast baseline; GRU, TCN and TFT are neural networks that can learn curves and longer patterns. Press “How it works” for a diagram."), archRow),
    h("div", { class: "field" }, step(2, "Generated data (select one or more)", "The generated runs the model learns from. They supply the power the model should predict. Wind and solar runs are trained separately."), kindSeg, h("div", { style: "height:12px" }), dsBox),
    h("div", { class: "field" }, step(3, "Lead time", "How many hours before the predicted hour the weather forecast was made. 12 to 36 hours is a day-ahead forecast, 0 to 6 hours is very short-term. It applies to every forecast model whose inputs are ticked."), leadBox),
    h("div", { class: "field" }, step(4, "Input features", "The weather inputs the model may use. Use the model filter in the list to pick inputs from one forecast model, or tick inputs from several to combine them. Past power values add the plant’s own earlier output as an input."), featBox),
    h("div", { class: "field" }, step(5, "Train / validation split", "Training uses data up to the date you pick. The days after it, up to the validation end date, test the model on data it has not seen. “By share” uses the last part of the data instead."), splitCard),
    h("div", { class: "field" }, step(6, "Model name (optional)", "A name to find the model later in the Saved models list. If you leave it empty, a name is made for you."), nameIn),
    h("div", { class: "field" }, step(7, "Settings", "How the model is trained: input window, forecast horizon, network size, learning rate and early stopping. The defaults are a good starting point."), hpGrid),
      h("div", { style: "margin-top:16px;display:flex;justify-content:center" }, trainBtn));

  // ------------------------------------------------------------ 2. models list
  const details = h("div", { class: "ml-details" });
  async function showDetails(id) {
    details.replaceChildren(h("div", { class: "note" }, "Loading…"));
    try {
      const m = await api.mlModel(id);
      const v = m.metrics.val;
      const kp = [["RMSE", v.rmse_pct, "%", 2], ["MAE", v.mae_pct, "%", 2], ["R²", v.r2, "", 3], ["Bias", v.bias_pct, "%", 2],
        ["Training windows", m.n_train, "", 0], ["Validation windows", m.n_val, "", 0]];
      if (m.metrics.zero_shot) kp.unshift(["RMSE before fine-tuning", m.metrics.zero_shot.rmse_pct, "%", 2]);
      const figs = {};
      for (const name of ["loss", "predictions", "horizon", "scatter", "residuals", ...(m.stage === "finetuned" ? ["comparison"] : [])]) {
        try { figs[name] = await api.mlFigure(id, name); } catch (_) { /* optional */ }
      }
      details.replaceChildren(
        h("h3", {}, m.name),
        h("div", { class: "kpis" }, kp.map(([l, val, u, d]) =>
          h("div", { class: "kpi" }, h("div", { class: "l" }, l), h("div", { class: "v" }, fmt(val, d), u ? h("span", { class: "u" }, u) : null)))),
        plotCards(figs));
      details.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (e) { details.replaceChildren(); toast(e.message, "error"); }
  }

  // One card per model: name + hover details, badges, headline error, actions.
  const modelCards = models.map((m) => {
    const rmse = m.metrics && m.metrics.val ? `${fmt(m.metrics.val.rmse_pct, 2)} %` : "–";
    return h("div", { class: "model-card" },
      h("div", { class: "model-cell" },
        h("div", { class: "model-name" }, m.name, h("span", { class: "info-dot", "aria-label": "Model details" }, "i")),
        summaryPop(m, archs)),
      h("div", { class: "run-meta" },
        h("span", { class: "kind-badge wind-arch" }, archs.architectures.find((a) => a.id === m.architecture)?.label || m.architecture.toUpperCase()),
        h("span", { class: `kind-badge ${m.kind}` }, m.kind),
        h("span", { class: "kind-badge" }, m.stage === "finetuned" ? "Fine-tuned" : "Pre-trained"),
        new Date(m.created).toLocaleString()),
      h("div", { class: "model-score" }, h("span", { class: "l" }, "Validation RMSE"), h("span", { class: "v" }, rmse)),
      h("div", { class: "model-actions" },
        h("button", { class: "btn small", onclick: () => showDetails(m.id) }, "View"),
        h("button", { class: "btn small", disabled: m.stage === "finetuned", title: m.stage === "finetuned" ? "Fine-tune the pre-trained parent model" : "", onclick: () => openFinetune(m) }, "Fine-tune"),
        h("a", { class: "btn small", href: `${window.BASE_PATH || ""}/api/ml/models/${m.id}/download`, style: "text-decoration:none" }, "Download"),
        h("button", { class: "btn small danger", onclick: async () => {
          if (!confirm(tr(`Delete model "${m.name}"? This cannot be undone.`))) return;
          try { await api.mlDelete(m.id); toast("Model deleted", "ok"); reload(); } catch (e) { toast(e.message, "error"); }
        } }, "Delete")));
  });
  const listCard = h("div", { class: "card ml-card run-picker" },
    h("div", { class: "run-picker-head" }, h("h3", {}, "Saved models"),
      h("p", {}, "The list of pre-trained models.")),
    models.length ? h("div", { class: "model-grid" }, modelCards)
      : h("p", { class: "hint", style: "text-align:center" }, "No models yet. Pre-train one in the Pre-train tab."));

  // ------------------------------------------------------------ 3. fine-tune panel
  const ftHost = h("div");
  function openFinetune(parent) {
    const sources = results.filter((r) => r.kind === parent.kind);
    if (!sources.length) return toast(`Generate a ${parent.kind} run for your site first. It supplies the weather inputs.`, "error");
    const src = h("select", {}, sources.map((r) => h("option", { value: r.id }, rl(r))));
    const unit = h("select", {}, ["W", "kW", "MW"].map((u) => h("option", { value: u }, u)));
    const picker = filePicker(".csv,.txt,text/csv");
    const cap = numField("Rated capacity (W, optional)", "", { min: 1 });
    const ep = numField("Epochs", 15, { min: 1, max: 300, step: 1 });
    const lr = numField("Learning rate", 0.0003, { min: 0.000001, max: 1, step: 0.0001 });
    const freeze = h("input", { type: "checkbox" });
    const nm = h("input", { placeholder: `${parent.name} → real` });
    const ftDate = h("input", { type: "date" });
    const go = h("button", { class: "btn primary" }, "Fine-tune on my data");
    const isLinear = parent.architecture === "linear";
    const alphaF = numField("Shrinkage towards pre-trained weights", 10, { min: 0.0000001, step: 1 });
    go.addEventListener("click", async () => {
      if (!picker.input.files.length) return toast("Choose your plant's CSV file first.", "error");
      go.disabled = true;
      try {
        const file = picker.input.files[0];
        const body = { feature_result_id: src.value, csv_text: await file.text(), filename: file.name, unit: unit.value,
          name: nm.value.trim() || null, capacity_w: cap.input.value ? cap.get() : null, train_until: ftDate.value || null,
          ...(isLinear ? { ridge_alpha: alphaF.get() } : { epochs: ep.get(), lr: lr.get(), freeze_backbone: freeze.checked }) };
        trackJob(await api.mlFinetune(parent.id, body), { title: `Fine-tuning ${parent.name}` });
        toast("Fine-tuning started. Progress is shown in the corner.", "ok");
        ftHost.replaceChildren();
      } catch (e) { toast(e.message, "error"); } finally { go.disabled = false; }
    });
    ftHost.replaceChildren(h("div", { class: "card ml-card run-picker" },
      h("h3", {}, `Fine-tune “${parent.name}” on your real data`),
      h("p", { class: "hint" }, "Pick a generated run for the same site (it sets the location; the weather is the forecast model the model was pre-trained on) and upload your measured power file."),
      h("div", { class: "form-card", style: "box-shadow:none;border:0;padding:0;grid-template-columns:2fr .7fr 1.6fr" },
        h("div", { class: "field" }, h("label", {}, "Weather source (generated run)"), src),
        h("div", { class: "field" }, h("label", {}, "Power unit in file"), unit),
        h("div", { class: "field" }, h("label", {}, "Plant data file"), picker.el)),
      h("div", { class: "hp-grid", style: "margin-top:12px" }, cap.el,
        isLinear ? alphaF.el : ep.el,
        ...(isLinear ? [] : [lr.el, h("label", { class: "check", style: "align-self:end" }, freeze, "Train only the output layer")]),
        h("div", { class: "field" }, h("label", {}, "Train until (optional, else last 20 % validates)"), ftDate),
        h("div", { class: "field" }, h("label", {}, "New model name (optional)"), nm)),
      h("div", { style: "margin-top:16px;display:flex;gap:8px;justify-content:center" }, go,
        h("button", { class: "btn", onclick: () => ftHost.replaceChildren() }, "Cancel"))));
    ftHost.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  const paneTrain = h("div", { class: "ml-pane", style: "max-width:1100px" }, trainCard);
  const paneModels = h("div", { class: "ml-pane", style: "max-width:1100px" }, listCard, ftHost, details);
  const tabs = [["train", "Pre-train", paneTrain], ["models", `Models (${models.length})`, paneModels]];
  const seg = h("div", { class: "seg" });
  function showTab(t) {
    activeTab = t;
    seg.replaceChildren(...tabs.map(([k, label]) => h("button", { class: k === t ? "active" : "", onclick: () => showTab(k) }, label)));
    tabs.forEach(([k, , pane]) => { pane.hidden = k !== t; });
  }
  // When a job finishes while this view is on screen, refresh and open the new model; otherwise the pop-up offers "View model".
  setFinishHandler((job) => {
    if (!container.isConnected) return false;
    activeTab = "models";
    pendingResult = job.result_id;
    toast("Model trained and saved", "ok");
    reload();
    return true;
  });
  const pending = takePending() || pendingResult;
  pendingResult = null;
  if (pending) { activeTab = "models"; }
  showTab(activeTab);

  container.replaceChildren(h("div", { class: "compare" },
    h("div", { class: "ml-head" }, h("div"), seg, h("div", { style: "justify-self:end" }, help)),
    !archs.torch ? h("div", { class: "ml-banner", style: "max-width:1400px;margin:0 auto 14px" }, "PyTorch isn't installed, so only Linear regression can be trained. Run `pip install torch` and restart the server to enable GRU, TCN and TFT.") : null,
    paneTrain, paneModels));
  if (pending) showDetails(pending);
}
