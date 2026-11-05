// "How it works" pop-up for the architecture cards in the Pre-train form.
// Draws a layer diagram from the current settings and estimates the number of trainable parameters.
import { h, fmt } from "./ui.js?v=26";

const GRN = (d) => 4 * d * d + 6 * d;                       // gated residual network: 2 dense + gate + layer norm

// Trainable parameters, mirroring the PyTorch / numpy definitions in mlmodels.py.
export function paramCount(arch, { L, F, H, layers, out }) {
  if (arch === "linear") return L * F * out + out;
  if (arch === "gru") {
    let n = 3 * (F * H + H * H + 2 * H);
    for (let i = 1; i < layers; i++) n += 3 * (2 * H * H + 2 * H);
    return n + H * out + out;
  }
  if (arch === "tcn") {
    let n = 0, cin = F;
    for (let i = 0; i < layers; i++) {
      n += cin * H * 3 + H + H * H * 3 + H + (cin !== H ? cin * H + H : 0);
      cin = H;
    }
    return n + H * out + out;
  }
  // compact TFT
  return F * 2 * H + F * F + F + 3 * GRN(H) + (8 * H * H + 8 * H) + (4 * H * H + 4 * H) + 2 * H + H * out + out;
}

function steps(arch, { L, F, H, layers, out }) {
  const input = ["Input", `${L} hours × ${F} features`];
  const output = ["Output", out === 1 ? "next-hour power (%)" : `power for the next ${out} hours (%)`];
  if (arch === "linear") return [input, ["Flatten", `${L * F} values in a row`], ["Weighted sum", `one weight per value → ${out} output${out > 1 ? "s" : ""}`], output];
  if (arch === "gru") return [input, [`GRU × ${layers}`, `${H} units, reads hour by hour`], ["Last hour's memory", `${H} values`], ["Dense layer", `→ ${out} output${out > 1 ? "s" : ""}`], output];
  if (arch === "tcn") return [input, [`Conv block × ${layers}`, `${H} channels, steps ${Array.from({ length: layers }, (_, i) => 2 ** i).join(", ")} h apart`], ["Latest time step", `${H} values`], ["Dense layer", `→ ${out} output${out > 1 ? "s" : ""}`], output];
  return [input, ["Input selection", "learns a weight for each feature"], ["Gated residual block", `${H} dims`], ["LSTM", `${H} units, short-term memory`],
    ["Self-attention", "4 heads, looks back at key hours"], ["Gated residual block", `${H} dims`], ["Dense layer", `→ ${out} output${out > 1 ? "s" : ""}`], output];
}

const WHEN = {
  linear: "Best as a quick baseline. Trains in seconds and is easy to interpret, but can't model curves such as a turbine's flat top.",
  gru: "A solid all-rounder for hourly weather sequences. Moderate training time.",
  tcn: "Good when longer patterns matter (daily cycles). Usually trains faster than a GRU.",
  tft: "The most flexible and usually most accurate, but the slowest to train. Works best with plenty of data.",
};

export function openArchPopup(info, cfg) {
  const rows = steps(info.id, cfg);
  const flow = h("div", { class: "arch-flow" }, rows.flatMap(([t, s], i) => [
    i ? h("div", { class: "arch-arrow" }, "↓") : null,
    h("div", { class: `arch-step${i === 0 || i === rows.length - 1 ? " io" : ""}` }, h("b", {}, t), h("span", {}, s)),
  ]));
  const close = () => { overlay.remove(); document.removeEventListener("keydown", onKey); };
  const onKey = (e) => { if (e.key === "Escape") close(); };
  const overlay = h("div", { class: "arch-overlay", onclick: (e) => { if (e.target === overlay) close(); } },
    h("div", { class: "arch-modal", role: "dialog", "aria-label": `${info.label} architecture` },
      h("button", { class: "arch-close", type: "button", "aria-label": "Close", onclick: close }, "×"),
      h("h3", {}, info.label),
      h("p", { class: "arch-lead" }, info.description),
      flow,
      h("div", { class: "arch-stats" },
        h("div", {}, h("b", {}, `≈ ${fmt(paramCount(info.id, cfg), 0)}`), h("span", {}, "trainable parameters")),
        h("div", {}, h("b", {}, `${cfg.F}`), h("span", {}, "input features")),
        h("div", {}, h("b", {}, `${cfg.L} h`), h("span", {}, "input window"))),
      h("p", { class: "arch-when" }, h("b", {}, "When to use it: "), WHEN[info.id] || ""),
      h("p", { class: "arch-note" }, "The diagram follows your current settings below; change them and reopen to see the difference.")));
  document.addEventListener("keydown", onKey);
  document.body.append(overlay);
}
