// Language switch (English / German). English is the source text of the app; German is looked up in lang_de.js.
// Everything that reaches the page is translated in one place: a DOM pass for text and tooltips, a deep pass for the
// Plotly figures the server sends, and tr() for browser dialogs. The choice is stored in the browser and applied by a reload.
import { DE, DE_PATTERNS } from "./lang_de.js?v=26";

let stored = null;
try { stored = localStorage.getItem("lang"); } catch (_) { /* storage unavailable */ }
export const lang = stored === "de" ? "de" : "en";
export const isDe = lang === "de";

export function setLang(next) {
  try { localStorage.setItem("lang", next); } catch (_) { /* storage unavailable */ }
  location.reload();
}

const cache = new Map();

function lookup(core) {
  if (Object.prototype.hasOwnProperty.call(DE, core)) return DE[core];
  const bold = /^<b>([\s\S]*)<\/b>$/.exec(core);                 // chart titles are wrapped in <b>
  if (bold) return `<b>${lookup(bold[1])}</b>`;
  if (core.includes(" · ")) {                                    // composite lines ("a · b · c"): translate every part
    return core.split(" · ").map(lookup).join(" · ");
  }
  for (const [re, to] of DE_PATTERNS) {
    const m = re.exec(core);
    if (!m) continue;
    // A function gets lookup() first, so it can translate the parts it captured (names, units, ...).
    return typeof to === "function" ? to(lookup, ...m.slice(1)) : core.replace(re, to);
  }
  return core;
}

/** Translate one text (keeps leading / trailing spaces). English stays as it is. */
export function tr(text) {
  if (!isDe || typeof text !== "string") return text;
  const m = /^(\s*)([\s\S]*?)(\s*)$/.exec(text);
  if (!m[2]) return text;
  let out = cache.get(m[2]);
  if (out === undefined) { out = lookup(m[2]); cache.set(m[2], out); }
  return m[1] + out + m[3];
}

/** Translate every string of a JSON structure (Plotly figures from the server). */
export function trDeep(v) {
  if (!isDe) return v;
  if (typeof v === "string") return tr(v);
  if (Array.isArray(v)) return typeof v[0] === "number" ? v : v.map(trDeep);
  if (v && typeof v === "object") {
    const o = {};
    for (const k of Object.keys(v)) o[k] = trDeep(v[k]);
    return o;
  }
  return v;
}

// ---------------------------------------------------------------- DOM
const ATTRS = ["title", "placeholder", "aria-label", "data-tip"];
const SKIP = new Set(["SCRIPT", "STYLE", "TEXTAREA", "NOSCRIPT"]);

function translateNode(node) {
  if (node.nodeType === Node.TEXT_NODE) {
    const next = tr(node.nodeValue);
    if (next !== node.nodeValue) node.nodeValue = next;
    return;
  }
  if (node.nodeType !== Node.ELEMENT_NODE || SKIP.has(node.tagName)) return;
  if (node.classList && (node.classList.contains("js-plotly-plot") || node.classList.contains("no-i18n"))) return;
  for (const a of ATTRS) {
    if (node.hasAttribute(a)) {
      const cur = node.getAttribute(a), next = tr(cur);
      if (next !== cur) node.setAttribute(a, next);
    }
  }
  for (const child of node.childNodes) translateNode(child);
}

export function initI18n() {
  document.documentElement.lang = lang;
  document.querySelectorAll("[data-lang]").forEach((b) => {
    b.classList.toggle("active", b.dataset.lang === lang);
    b.addEventListener("click", () => { if (b.dataset.lang !== lang) setLang(b.dataset.lang); });
  });
  if (!isDe) return;
  if (window.Plotly && Plotly.setPlotConfig) Plotly.setPlotConfig({ locale: "de" });     // German month names, modebar texts
  document.title = tr(document.title);
  translateNode(document.body);
  new MutationObserver((records) => {
    for (const r of records) {
      if (r.type === "childList") r.addedNodes.forEach(translateNode);
      else if (r.type === "characterData") translateNode(r.target);
      else if (r.type === "attributes") translateNode(r.target);
    }
  }).observe(document.body, { subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ATTRS });
}
