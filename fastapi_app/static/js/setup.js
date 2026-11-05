// Setup view: Leaflet map + docked inspector. Edits autosave to the scenario.
import { api } from "./api.js?v=26";
import { h, toast, debounce } from "./ui.js?v=26";

const BASEMAPS = {
  Streets: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png", "© OpenStreetMap"],
  Light: ["https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png", "© OpenStreetMap © CARTO"],
  Satellite: ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", "© Esri"],
};

const DEFAULT_COMMISSIONING = "2013-01-01";

export function mountSetup(container, scenario, boot, onChange) {
  const params = scenario.params;
  if (scenario.kind === "wind" && params.annual_degradation == null) params.annual_degradation = boot.wind_defaults.annual_degradation;
  const kind = scenario.kind;
  if (kind === "wind") params.turbines.forEach((t) => { if (!t.commissioning_date) t.commissioning_date = DEFAULT_COMMISSIONING; });
  const state = { selected: null, addMode: false, specs: {} };

  const mapEl = h("div", { id: "map" });
  const mapWrap = h("div", { class: "map-wrap" }, mapEl);
  const inspector = h("aside", { class: "inspector" });
  container.replaceChildren(h("div", { class: "setup" }, mapWrap, inspector));

  const save = debounce(async () => {
    try { await api.updateScenario(scenario.id, { params }); onChange && onChange(); }
    catch (e) { toast(`Save failed: ${e.message}`, "error"); }
  }, 600);

  // ---------------------------------------------------------------- map
  const map = L.map(mapEl, { zoomControl: true }).setView([51.2, 10.4], 6);
  const bases = {};
  for (const [name, [url, attr]] of Object.entries(BASEMAPS)) bases[name] = L.tileLayer(url, { attribution: attr, maxZoom: 19 });
  bases.Streets.addTo(map);
  const gridLayer = L.layerGroup();
  const stationLayer = L.layerGroup();
  const turbineLayer = L.layerGroup().addTo(map);
  const overlays = kind === "wind" ? { "NWP grid points": gridLayer, Turbines: turbineLayer } : { "Solar stations": stationLayer };
  L.control.layers(bases, overlays, { position: "topright" }).addTo(map);

  // search box
  const searchInput = h("input", { placeholder: "Search a place…" });
  const results = h("div", { class: "search-results", hidden: true });
  const doSearch = async () => {
    const q = searchInput.value.trim();
    if (!q) return;
    try {
      const { results: rs } = await api.search(q);
      results.replaceChildren(...(rs.length ? rs : [{ display_name: "No results found (Germany only)", none: true }]).map((r) =>
        h("div", { onclick: () => { if (!r.none) { flyTo(r.lat, r.lon, 11); results.hidden = true; } } }, r.display_name)));
      results.hidden = false;
    } catch (e) { toast(e.message, "error"); }
  };
  searchInput.addEventListener("keydown", (e) => { if (e.key === "Enter") doSearch(); });
  const tools = h("div", { class: "map-tools" },
    h("div", { class: "search", style: "position:relative" }, searchInput, h("button", { onclick: doSearch }, "🔍"), results));
  mapWrap.append(tools);
  L.DomEvent.disableClickPropagation(tools);
  L.DomEvent.disableScrollPropagation(tools);

  // Keep the target visible next to the docked inspector (it is outside the map, so plain flyTo works).
  const flyTo = (lat, lon, z) => map.flyTo([lat, lon], z || map.getZoom(), { duration: 0.6 });

  async function loadGrid() {
    if (kind !== "wind" || !map.hasLayer(gridLayer) || map.getZoom() < 8) { gridLayer.clearLayers(); return; }
    const b = map.getBounds();
    try {
      const { points } = await api.gridPoints({ south: b.getSouth(), west: b.getWest(), north: b.getNorth(), east: b.getEast() });
      gridLayer.clearLayers();
      points.forEach(([la, lo]) => L.circleMarker([la, lo], { radius: 3, color: "#64748b", weight: 1, fillOpacity: 0.5 }).addTo(gridLayer));
    } catch (_) { /* non-critical */ }
  }
  map.on("moveend", debounce(loadGrid, 300));
  map.on("overlayadd overlayremove", loadGrid);

  if (kind === "solar") {
    boot.solar_stations.forEach((s) =>
      L.circleMarker([s.lat, s.lon], { radius: 4, color: "#f59e0b", weight: 1, fillOpacity: 0.7 })
        .bindTooltip(`Station ${s.id}`).addTo(stationLayer));
    stationLayer.addTo(map);
  }

  // ---------------------------------------------------------------- markers
  let solarMarker = null;
  // Solar marker: a small array of tilted panels (2 rows x 3); the map position is the centre of the array.
  function solarIcon() {
    const size = 84;
    const panel = (x, y) => `<rect x="${x}" y="${y}" width="23" height="20" rx="1.5"/>`;
    const cells = [18, 43, 68].flatMap((x) => [panel(x, 22), panel(x, 47)]).join("");
    const svg = `<svg viewBox="0 0 100 100" width="${size}" height="${size}" aria-hidden="true">
      <g class="s-panels" transform="translate(8 0) skewX(-18)">${cells}</g>
      <g class="s-grid" transform="translate(8 0) skewX(-18)"><path d="M18 34 H91 M18 59 H91 M41 22 V67 M66 22 V67"/></g></svg>`;
    return L.divIcon({ className: "", iconSize: [size, size], iconAnchor: [size / 2, size * 0.45], tooltipAnchor: [0, -size * 0.25],
      html: `<div class="solar-mark" style="width:${size}px;height:${size}px">${svg}</div>` });
  }
  function drawSolar() {
    if (solarMarker) map.removeLayer(solarMarker);
    solarMarker = null;
    if (params.lat == null) return;
    solarMarker = L.marker([params.lat, params.lon], { draggable: true, icon: solarIcon() })
      .bindTooltip("Solar plant (drag to move)").addTo(map);
    solarMarker.on("dragend", () => { const p = solarMarker.getLatLng(); setSolar(p.lat, p.lng); });
  }
  async function setSolar(lat, lon) {
    params.lat = +lat.toFixed(5); params.lon = +lon.toFixed(5);
    drawSolar(); renderInspector(); save();
  }

  // Turbine marker: a small wind turbine (tower + three blades), sized by the model's rotor diameter.
  // The map position is the tower base. Rotor diameters are fetched once per model and cached.
  const diameters = {};
  function turbineSize(model) {
    const d = diameters[model];
    return d ? Math.round(Math.min(104, Math.max(64, d / 1.4))) : 72;
  }
  function loadDiameter(model) {
    if (model in diameters) return;
    diameters[model] = null;
    api.turbineSpecs(model).then((sp) => { diameters[model] = sp.diameter || null; drawTurbines(); }).catch(() => {});
  }
  function pinIcon(i, sel, model) {
    const size = turbineSize(model);
    const blade = (deg) => `<path transform="rotate(${deg} 50 40)" d="M50 40 C45 30 46.5 14 50 4 C53.5 14 55 30 50 40Z"/>`;
    const svg = `<svg viewBox="0 0 100 100" width="${size}" height="${size}" aria-hidden="true">
      <path class="t-tower" d="M47.8 42 L52.2 42 L54 95 L46 95Z"/>
      <g class="t-rotor">${blade(0)}${blade(120)}${blade(240)}<circle cx="50" cy="40" r="4.6" class="t-hub" /></g></svg>`;
    return L.divIcon({ className: "", iconSize: [size, size], iconAnchor: [size / 2, size * 0.94], tooltipAnchor: [0, -size * 0.8],
      html: `<div class="turbine-mark${sel ? " sel" : ""}" style="width:${size}px;height:${size}px">${svg}<span class="t-num">${i + 1}</span></div>` });
  }
  function drawTurbines() {
    turbineLayer.clearLayers();
    if (kind !== "wind") return;
    params.turbines.forEach((t, i) => {
      loadDiameter(t.turbine);
      const m = L.marker([t.lat, t.lon], { draggable: true, icon: pinIcon(i, state.selected === i, t.turbine) })
        .bindTooltip(`#${i + 1} ${t.turbine}`).addTo(turbineLayer);
      m.on("click", () => { state.selected = i; drawTurbines(); renderInspector(); });
      m.on("dragend", () => { const p = m.getLatLng(); t.lat = +p.lat.toFixed(5); t.lon = +p.lng.toFixed(5); renderInspector(); save(); });
    });
  }
  function fitAll() {
    const pts = kind === "wind" ? params.turbines.map((t) => [t.lat, t.lon]) : params.lat != null ? [[params.lat, params.lon]] : [];
    if (pts.length > 1) map.fitBounds(pts, { padding: [60, 60], maxZoom: 13 });
    else if (pts.length === 1) map.setView(pts[0], 11);
  }

  map.on("click", (e) => {
    if (kind === "solar") return setSolar(e.latlng.lat, e.latlng.lng);
    if (!state.addMode) return;
    addTurbine(e.latlng.lat, e.latlng.lng);
  });

  // ---------------------------------------------------------------- wind actions
  const defaults = { turbine: boot.turbine_types[0], hub_height: 100 };
  function addTurbine(lat, lon, extra = {}) {
    params.turbines.push({ lat: +lat.toFixed(5), lon: +lon.toFixed(5), turbine: defaults.turbine,
      hub_height: defaults.hub_height, commissioning_date: DEFAULT_COMMISSIONING, ...extra });
    state.selected = params.turbines.length - 1;
    drawTurbines(); renderInspector(); save();
  }

  // ---------------------------------------------------------------- inspector
  function field(label, input, cls = "") { return h("div", { class: `field ${cls}`.trim() }, h("label", {}, label), input); }
  function num(obj, key, opts = {}, after) {
    const i = h("input", { type: "number", step: opts.step ?? "any", min: opts.min, max: opts.max, value: obj[key] ?? "" });
    i.addEventListener("change", () => { obj[key] = i.value === "" ? null : Number(i.value); after && after(); save(); });
    return i;
  }
  function check(label, obj, key, after) {
    const i = h("input", { type: "checkbox" }); i.checked = !!obj[key];
    i.addEventListener("change", () => { obj[key] = i.checked; after && after(); save(); });
    return h("label", { class: "check" }, i, label);
  }
  function turbineSelect(value, onPick) {
    const s = h("select", {}, boot.turbine_types.map((n) => h("option", { value: n, selected: n === value }, n)));
    s.addEventListener("change", () => onPick(s.value));
    return s;
  }

  function renderInspector() {
    inspector.replaceChildren(...(kind === "wind" ? windInspector() : solarInspector()));
  }

  function solarInspector() {
    return [
      h("div", { class: "sect" }, h("h3", {}, "Location"),
        params.lat == null ? h("p", { class: "hint" }, "Click the map to place the plant. Drag the pin to adjust.")
          : h("p", { class: "hint" }, `${params.lat.toFixed(4)}, ${params.lon.toFixed(4)}`)),
      h("div", { class: "sect" }, h("h3", {}, "PV system"),
        field("DC rating (W)", num(params, "dc_rating_watts", { min: 1 })),
        h("div", { class: "row" }, field("Tilt (°)", num(params, "surface_tilt", { min: 0, max: 90 })),
          field("Azimuth (°)", num(params, "surface_azimuth", { min: 0, max: 360 }))),
        h("div", { class: "row" }, field("γ Pdc (1/°C)", num(params, "gamma_pdc")), field("Albedo", num(params, "albedo", { min: 0, max: 1 }))),
        field("Inverter efficiency", num(params, "eta_inv_nom", { min: 0.5, max: 1 }))),
    ];
  }

  function windInspector() {
    const sel = state.selected != null ? params.turbines[state.selected] : null;
    const list = h("ul", { class: "turbine-list" }, params.turbines.map((t, i) =>
      h("li", { class: `turbine-item${i === state.selected ? " active" : ""}`, onclick: () => { state.selected = i; drawTurbines(); flyTo(t.lat, t.lon); renderInspector(); } },
        h("span", { class: "badge" }, i + 1),
        h("div", { class: "t" }, h("b", {}, t.turbine), `${t.lat.toFixed(3)}, ${t.lon.toFixed(3)} · ${t.hub_height} m`),
        h("button", { title: "Remove", onclick: (e) => {
          e.stopPropagation(); params.turbines.splice(i, 1);
          state.selected = null; drawTurbines(); renderInspector(); save();
        } }, "✕"))));

    const addBtn = h("button", { class: `btn small${state.addMode ? " primary" : ""}`, onclick: () => {
      state.addMode = !state.addMode; mapEl.style.cursor = state.addMode ? "crosshair" : ""; renderInspector();
    } }, state.addMode ? "Click map to place… (done)" : "+ Place turbines on map");

    const parts = [
      h("div", { class: "sect" }, h("h3", {}, `Turbines (${params.turbines.length})`),
        h("div", { class: "row" }, field("Turbine Type", turbineSelect(defaults.turbine, (v) => { defaults.turbine = v; }), "center"),
          field("Hub height (m)", (() => { const i = h("input", { type: "number", value: defaults.hub_height, min: 10 }); i.addEventListener("change", () => { defaults.hub_height = Number(i.value) || 100; }); return i; })(), "center")),
        addBtn,
        params.turbines.length ? list : h("p", { class: "hint" }, "No turbines yet. Place them on the map.")),
    ];

    if (sel) {
      parts.push(h("div", { class: "sect" }, h("h3", {}, `Turbine ${state.selected + 1}`),
        field("Model", turbineSelect(sel.turbine, (v) => { sel.turbine = v; drawTurbines(); renderInspector(); save(); }), "center"),
        h("div", { class: "row" }, field("Hub height (m)", num(sel, "hub_height", { min: 10 }), "center"),
          field("Commissioned", (() => { const i = h("input", { type: "date", value: sel.commissioning_date || "" }); i.addEventListener("change", () => { sel.commissioning_date = i.value || null; save(); }); return i; })())),
        h("div", { class: "row" }, field("Latitude", num(sel, "lat", { min: -90, max: 90 }, drawTurbines)), field("Longitude", num(sel, "lon", { min: -180, max: 180 }, drawTurbines)))));
    }

    parts.push(h("div", { class: "sect" }, h("h3", {}, "Losses"),
      check("Wake losses (Jensen/Park)", params, "apply_wake", renderInspector),
      params.apply_wake ? field(`Wake decay constant (k)`, num(params, "wake_decay", { min: 0.01, max: 0.3, step: 0.005 })) : null,
      check("Ageing degradation", params, "apply_ageing", renderInspector),
      params.apply_ageing ? field(`Annual degradation`, num(params, "annual_degradation", { min: 0, max: 0.05, step: 0.0005 }, renderInspector)) : null));
    return parts;
  }

  drawSolar(); drawTurbines(); renderInspector(); fitAll();
  setTimeout(() => map.invalidateSize(), 50);
  return { destroy: () => map.remove() };
}
