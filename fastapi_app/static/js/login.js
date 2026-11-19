import { initI18n, tr } from "./i18n.js?v=26";

initI18n();
const form = document.getElementById("login-form");
const err = document.getElementById("login-error");
const btn = document.getElementById("login-btn");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  err.hidden = true;
  btn.disabled = true;
  try {
    const BASE = window.BASE_PATH || "";
    const res = await fetch(`${BASE}/api/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: form.username.value, password: form.password.value }),
    });
    if (res.ok) { location.href = `${BASE}/`; return; }
    let msg = "Sign-in failed.";
    try { const d = await res.json(); if (typeof d.detail === "string") msg = d.detail; } catch (_) { /* no body */ }
    err.textContent = tr(msg);
    err.hidden = false;
  } catch (_) {
    err.textContent = tr("Could not reach the server.");
    err.hidden = false;
  } finally {
    btn.disabled = false;
  }
});
