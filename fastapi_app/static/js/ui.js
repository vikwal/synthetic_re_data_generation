// Small DOM helpers shared by all views.
export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "html") el.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (k === "value") el.value = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return el;
}

export function toast(message, type = "") {
  const el = h("div", { class: `toast ${type}` }, message);
  document.getElementById("toasts").append(el);
  setTimeout(() => el.remove(), type === "error" ? 7000 : 3500);
}

export const debounce = (fn, ms = 400) => {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
};

export function fmt(v, digits = 1) {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  return Number(v).toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: 0 });
}

// Popup menu anchored to an element.
export function popMenu(anchor, items) {
  document.querySelectorAll(".menu-pop").forEach((m) => m.remove());
  const r = anchor.getBoundingClientRect();
  const menu = h("div", { class: "menu-pop", style: `left:${Math.min(r.left, innerWidth - 190)}px;top:${r.bottom + 4}px` },
    items.map((it) => h("button", {
      class: it.danger ? "danger" : "",
      onclick: () => { menu.remove(); it.run(); },
    }, it.label)));
  document.body.append(menu);
  const close = (e) => { if (!menu.contains(e.target)) { menu.remove(); document.removeEventListener("mousedown", close); } };
  setTimeout(() => document.addEventListener("mousedown", close), 0);
}
