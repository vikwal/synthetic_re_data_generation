// FL model tab: a "coming soon" page for the planned federated learning feature.
import { h } from "./ui.js?v=26";

// Network sketch: sites around a shared model. Dashes travel along the links; nodes breathe softly.
const ART = `
<svg viewBox="0 0 440 300" role="img" aria-label="Wind parks and solar plants connected to one shared model" class="fl-svg">
  <defs>
    <radialGradient id="fl-glow" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#059669" stop-opacity=".22"/><stop offset="100%" stop-color="#059669" stop-opacity="0"/>
    </radialGradient>
    <symbol id="fl-turbine" viewBox="0 0 40 40">
      <g fill="none" stroke="#059669" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M20 17 L20 37 M15 37 H25"/>
        <path d="M20 17 L20 3 M20 17 L32.1 24 M20 17 L7.9 24"/>
      </g>
      <circle cx="20" cy="17" r="2.6" fill="#059669"/>
    </symbol>
    <symbol id="fl-panel" viewBox="0 0 40 40">
      <g fill="none" stroke="#059669" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M9 10 H35 L31 26 H5 Z" fill="#d1fae5"/>
        <path d="M7 18 H33 M17.7 10 L13.7 26 M26.3 10 L22.3 26"/>
        <path d="M20 26 V34 M13 34 H27"/>
      </g>
    </symbol>
  </defs>
  <circle cx="220" cy="150" r="120" fill="url(#fl-glow)"/>
  <g class="fl-links" fill="none" stroke="#059669" stroke-width="1.6" stroke-linecap="round">
    <path d="M220 150 L70 62"/><path d="M220 150 L58 170"/><path d="M220 150 L110 262"/>
    <path d="M220 150 L370 62"/><path d="M220 150 L382 170"/><path d="M220 150 L330 262"/>
  </g>
  <g class="fl-sites" font-size="20" text-anchor="middle">
    <g class="fl-node" style="--d:0s"><circle cx="70" cy="62" r="26"/><use href="#fl-turbine" x="52" y="44" width="36" height="36"/></g>
    <g class="fl-node" style="--d:.5s"><circle cx="58" cy="170" r="26"/><use href="#fl-panel" x="40" y="152" width="36" height="36"/></g>
    <g class="fl-node" style="--d:1s"><circle cx="110" cy="262" r="26"/><use href="#fl-turbine" x="92" y="244" width="36" height="36"/></g>
    <g class="fl-node" style="--d:.25s"><circle cx="370" cy="62" r="26"/><use href="#fl-panel" x="352" y="44" width="36" height="36"/></g>
    <g class="fl-node" style="--d:.75s"><circle cx="382" cy="170" r="26"/><use href="#fl-turbine" x="364" y="152" width="36" height="36"/></g>
    <g class="fl-node" style="--d:1.25s"><circle cx="330" cy="262" r="26"/><use href="#fl-panel" x="312" y="244" width="36" height="36"/></g>
  </g>
  <g class="fl-hub">
    <circle class="fl-ring" cx="220" cy="150" r="46"/>
    <circle cx="220" cy="150" r="38" fill="#059669"/>
    <text x="220" y="146" text-anchor="middle" fill="#fff" font-size="11" font-weight="700" letter-spacing=".06em">SHARED</text>
    <text x="220" y="162" text-anchor="middle" fill="#fff" font-size="11" font-weight="700" letter-spacing=".06em">MODEL</text>
  </g>
</svg>`;

export function mountFL(container) {
  const art = h("div", { class: "fl-art" });
  art.innerHTML = ART;                                      // static, trusted markup
  container.replaceChildren(h("div", { class: "fl-page" },
    h("div", { class: "card fl-hero" },
      h("div", { class: "fl-copy" },
        h("span", { class: "fl-badge" }, h("i", { class: "fl-dot" }), "Coming soon"),
        h("h2", {}, "Federated Learning"),
        h("p", { class: "fl-lead" }, "Train one shared forecasting model across many wind parks and solar plants, without moving anyone's raw data."),
        h("p", { class: "fl-note" }, "This feature is under development. It will appear here as soon as it is ready.")),
      art)));
}
