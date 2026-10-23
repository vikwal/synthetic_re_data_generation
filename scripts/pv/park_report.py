#!/usr/bin/env python
"""HTML report per park of the geometry measurement (window method, 2026-10-22): the content of the Trianel window
notebooks as one self-contained page per park, plus an index with filters.

Per park: register entry and assignment, OSM outline on the DOP20, grey image and 2-D spectrum with the peak, window
map of the orthophoto, surface source and survey year, height above ground with a cross-section, window grid, one
example window in detail (folded profile with the line fit, planes per row), maps of tilt / azimuth / pitch / terrain
slope per window with the rejected windows and their reason, groups and arrays, 3-D model of the example window,
comparison with the register class. Images are embedded as PNG (base64).

    synthre/bin/python scripts/pv/park_report.py fleet [--parks ID,ID] [--suffix _v2] [--out DIR]
    synthre/bin/python scripts/pv/park_report.py trianel [--pv pv1 ...]          # confidential, stays on l2
Fleet: rasters from the tile cache (geometry_release_states.fetch_tiles finds the cached tiles), windows from
results/pv/geodata_release<suffix>/<park>/, park rows from data/parks_pv_v1/geodata_release_<state><suffix>.csv.
Output: ~/Work/pv_sim2real/results/geo_reports_v2/<park>.html + index.html; trianel -> geo_reports_trianel/."""
import argparse
import base64
import glob
import html
import io
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pv import geodata_tiles as GT, geodata_windows as GW  # noqa: E402

PAPER = os.path.expanduser("~/Work/pv_sim2real")
OUT_FLEET = os.path.join(PAPER, "results", "geo_reports_v2")
OUT_TRIANEL = os.path.join(PAPER, "results", "geo_reports_trianel")
plt.rcParams.update({"figure.dpi": 80, "font.size": 9})
SHORT = {"g_pattern": "M", "g_contrast": "K", "g_flank": "F", "g_ridge": "H", "g_rows": "R", "g_est": "S", "zu wenig Tischzellen": "-", "anderer Park": "A", "anderer Park (Bauweise)": "B",
         "anderer Park (Fläche)": "C",
         "zu wenig Oberfläche": "O", "kein Profil": "P", "isoliert": "I"}
SHORT_TXT = ("M = kein Reihenmuster, K = Kontrast < 0.5 m, F = Flanke außerhalb 1.5 m … 0.75·Pitch, H = First außerhalb 0.8–4.5 m, "
             "R = < 2 Reihen, S = Schätzer weichen ab (3° Punktwolke / 5° Raster), O = zu wenig Oberfläche, P = kein Profil, "
             "I = isoliert, A = anderer Park, B = anderer Park mit anderer Bauweise (Kastenmodus), C = jenseits der Fläche zur Leistung")


# ------------------------------------------------------------------------------------------------ helpers
def png(fig) -> str:
    b = io.BytesIO()
    fig.savefig(b, format="png", bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(b.getvalue()).decode()}" style="max-width:100%">'


def table(df: pd.DataFrame, digits: int = 2) -> str:
    return df.round(digits).to_html(index=False, border=0, classes="t", na_rep="–", escape=True)


def read_csv0(path: str) -> pd.DataFrame:
    """CSV that may be empty (a park without groups or arrays writes a header-less file)."""
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def kv(d: dict) -> str:
    rows = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>" for k, v in d.items())
    return f'<table class="t kv">{rows}</table>'


def reason_code(r: str) -> str:
    if not r:
        return ""
    return "".join(SHORT.get(x, "?") for x in str(r).split(","))


# ------------------------------------------------------------------------------------------------ figures
def fig_outline(rgb, inside, extent, poly_xy, pts, title):
    fig, ax = plt.subplots(figsize=(7, 6.5))
    if rgb is not None:
        ax.imshow(np.moveaxis(np.clip(np.nan_to_num(rgb), 0, 255).astype(np.uint8), 0, -1), extent=extent)
    ax.contour(np.flipud(inside.astype(float)), levels=[0.5], colors="red", linewidths=0.9, extent=extent)
    for lab, (x, y), m in pts:
        ax.plot(x, y, m, ms=12, mew=2, label=lab)
    if pts:
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title); ax.set_xlabel("m Ost"); ax.set_ylabel("m Nord")
    return png(fig)


def fig_spectrum(rgb, inside, res, conv, rows):
    """Grey image (1 m) and its 2-D power spectrum in the pitch band (display of pv.imagery.row_spectrum)."""
    g = np.nanmean(rgb, axis=0); blk = 2
    H1, W1 = (g.shape[0] // blk) * blk, (g.shape[1] // blk) * blk
    g1 = np.nanmean(g[:H1, :W1].reshape(H1 // blk, blk, W1 // blk, blk), axis=(1, 3))
    v1 = inside[:H1, :W1].reshape(H1 // blk, blk, W1 // blk, blk).mean(axis=(1, 3)) > 0.99
    x = np.where(v1 & np.isfinite(g1), g1 - np.nanmean(g1[v1]), 0.0)
    F = np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2
    ky = np.fft.fftshift(np.fft.fftfreq(x.shape[0], d=1.0)); kx = np.fft.fftshift(np.fft.fftfreq(x.shape[1], d=1.0))
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    ax[0].imshow(np.where(v1, g1, np.nan), cmap="gray", extent=[0, W1 * res, 0, H1 * res]); ax[0].set_title("Graubild 1 m im Umring")
    ax[1].imshow(np.log10(F + 1), cmap="magma", extent=[kx[0], kx[-1], ky[-1], ky[0]]); ax[1].set_xlim(-0.3, 0.3); ax[1].set_ylim(0.3, -0.3)
    for r_ in (1 / 15, 1 / 4):
        ax[1].add_patch(plt.Circle((0, 0), r_, fill=False, color="w", lw=0.6, ls="--"))
    if rows["peaks"]:
        a, p, s = rows["peaks"][0]
        wv = np.radians((a + 90) % 180)                           # wave vector compass (axis + 90), image y points south
        for sgn in (1, -1):
            ax[1].plot(sgn * np.sin(wv) / p, -sgn * np.cos(wv) / p, "c+", ms=14, mew=2)
    ax[1].set_title("log10 Leistung, Band 4–15 m (Kreise), Peak (+)"); ax[1].set_xlabel("kx (1/m)"); ax[1].set_ylabel("ky")
    pk = pd.DataFrame(rows["peaks"], columns=["axis_grid", "pitch_m", "share"])
    if len(pk):
        ax[2].bar(range(len(pk)), pk.share, color="#6A4C93"); ax[2].set_xticks(range(len(pk)))
        ax[2].set_xticklabels([f"{(a + conv) % 180:.0f}° / {p:.1f} m" for a, p in zip(pk.axis_grid, pk.pitch_m)])
    ax[2].set_title("Peaks: Achse wahr / Pitch, Anteil der Bandleistung")
    return png(fig)


def fig_ortho_windows(rgb, extent, H, res, rows):
    loc = rows["windows"]
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.imshow(np.moveaxis(np.clip(np.nan_to_num(rgb), 0, 255).astype(np.uint8), 0, -1), extent=extent)
    if len(loc):
        sc = ax.scatter(loc[:, 0], H * res - loc[:, 1], c=loc[:, 2], cmap="hsv", vmin=0, vmax=180, s=55, edgecolor="k", lw=0.4)
        plt.colorbar(sc, ax=ax, label="Achse (° Gitter)", shrink=0.8)
    ax.set_title("Orthophoto: lokale Reihenachse je 80-m-Fenster (Startwert pitch0)")
    return png(fig)


def fig_surface(surf, dgm, hag, inside, extent, res, source, cut):
    lo, hi = (np.nanpercentile(dgm[inside], [2, 98]) if np.isfinite(dgm[inside]).any() else (0, 1))
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for a_, z, t, kw in [(ax[0], surf, f"Oberfläche ({source})", dict(cmap="terrain", vmin=lo, vmax=hi + 3)),
                         (ax[1], dgm, "Gelände (DGM1)", dict(cmap="terrain", vmin=lo, vmax=hi + 3)),
                         (ax[2], np.clip(hag, -0.5, 4), "Höhe über Grund (m)", dict(cmap="viridis", vmin=-0.5, vmax=4))]:
        im = a_.imshow(z, extent=extent, **kw); a_.set_title(t); plt.colorbar(im, ax=a_, shrink=0.7)
    if cut is not None:
        ax[2].plot(*cut["line"], "w--", lw=0.8)
    out = png(fig)
    if cut is not None:
        fig, axq = plt.subplots(figsize=(14, 3))
        axq.plot(cut["x"], cut["surf"], "o-", ms=2, lw=0.7, color="#6A4C93", label="Oberfläche")
        axq.plot(cut["x"], cut["dgm"], color="#444", lw=1.1, label="DGM1")
        axq.set_xlabel("m entlang des Schnitts quer zu den Reihen"); axq.set_ylabel("m ü. NN"); axq.legend()
        axq.set_title("Querschnitt 60 m durch das Beispielfenster (Sägezahn = Tischreihen)")
        out += png(fig)
    return out


def fig_grid(hag, win, nw, win_m, extent, H, res, ex):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(np.clip(hag, -0.5, 4), cmap="Greys", extent=extent, alpha=0.85)
    for r_ in win.itertuples():
        x0, y0 = r_.j0 * res, (H - r_.i0 - nw) * res
        tab = r_.f_tab >= GW.F_TAB_MIN
        ax.add_patch(plt.Rectangle((x0, y0), win_m, win_m, fill=tab, facecolor=(0.88, 0.48, 0.09, 0.22),
                                   edgecolor="#B3121A" if r_.w == ex else ("#1F4E99" if r_.ok else "#8A8A8A"), lw=2.0 if r_.w == ex else 0.7))
    ax.set_title(f"Fenstergitter {win_m:.0f} m (blau = angenommen, grau = verworfen, gefüllt = ≥ 15 % Tische, rot = Beispiel)")
    return png(fig)


def fig_window(det, res_w):
    cen, prof2, k_r, k_f, R = det["cen"], det["prof2"], det["k_r"], det["k_f"], det["R"]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
    ax[0].imshow(np.clip(det["hw"], -0.5, 4), cmap="viridis", vmin=-0.5, vmax=4,
                 extent=[det["Xw"].min(), det["Xw"].max(), det["Yw"].min(), det["Yw"].max()])
    ax[0].set_title(f"Fenster {res_w['w']}: Höhe über Grund, Tische {res_w['f_tab']:.0%}")
    ax[1].plot(cen, prof2, color="#6A4C93", lw=2)
    ax[1].plot([cen[k_r], cen[k_f]], [res_w["h_ridge"], res_w["h_foot"]], "o", color="#B3121A")
    ax[1].axhline(GW.MIN_H, color="k", ls=":", lw=0.8)
    kk = np.arange(min(k_r, k_f), max(k_r, k_f) + 1); inn = kk[1:-1] if len(kk) >= 5 else kk; inn = inn[np.isfinite(prof2[inn])]
    if len(inn) >= 3:
        cf = np.polyfit(cen[inn], prof2[inn], 1); ax[1].plot(cen[inn], np.polyval(cf, cen[inn]), "--", color="#E07B17", lw=1.6, label="Ausgleichsgerade")
        ax[1].legend()
    ax[1].set_title(f"gefaltetes Profil: Gerade {res_w['tilt_profile']:.1f}° (Endpunkte {res_w['tilt_profile_ends']:.1f}°), "
                    f"horizontal {res_w['tilt_horiz']:.1f}°, nach {res_w['azimuth_true']:.0f}°", fontsize=8.5)
    ax[1].set_xlabel("s mod Pitch (m)"); ax[1].set_ylabel("Median-Höhe (m)")
    if len(R):
        ax[2].bar(range(len(R)), R.tilt, color="#6A4C93"); ax[2].axhline(res_w["tilt_rows"], color="#B3121A")
    ax[2].set_title(f"{len(R)} Reihen-Ebenen, Median {res_w['tilt_rows']:.1f}° (gegen Boden)"); ax[2].set_xlabel("Reihe")
    return png(fig)


def fig_maps(hag, win, nw, win_m, extent, H, res):
    fig, axs = plt.subplots(2, 2, figsize=(15, 13))
    spec = [("tilt_horiz", "Neigung horizontal (°)", 0, 35, "viridis"), ("azimuth_true", "Azimut (°)", 0, 360, "hsv"),
            ("pitch", "Pitch (m)", 4, 16, "plasma"), ("slope_facing_deg", "Gefälle in Blickrichtung (°)", -6, 6, "coolwarm")]
    for a_, (col, lab, vmin, vmax, cmap) in zip(axs.ravel(), spec):
        a_.imshow(np.clip(hag, -0.5, 4), cmap="Greys", extent=extent, alpha=0.55)
        for r_ in win.itertuples():
            x0, y0 = r_.j0 * res, (H - r_.i0 - nw) * res
            if r_.ok and col in win:
                v = getattr(r_, col); c = plt.get_cmap(cmap)((v - vmin) / (vmax - vmin))
                a_.add_patch(plt.Rectangle((x0, y0), win_m, win_m, facecolor=(*c[:3], 0.78), edgecolor="k", lw=0.4))
                t = f"{round(v) % 360:.0f}" + ("" if r_.az_sure else "?") if col == "azimuth_true" else f"{v:.0f}" if col != "slope_facing_deg" else f"{v:+.1f}"
                a_.text(x0 + win_m / 2, y0 + win_m / 2, t, ha="center", va="center", fontsize=6.5)
            elif r_.f_tab >= GW.F_TAB_MIN:
                a_.add_patch(plt.Rectangle((x0, y0), win_m, win_m, facecolor=(0.5, 0.5, 0.5, 0.3), edgecolor="k", lw=0.3, hatch="//"))
                a_.text(x0 + win_m / 2, y0 + win_m / 2, reason_code(r_.reason), ha="center", va="center", fontsize=6.5, color="#7a0000")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax)); plt.colorbar(sm, ax=a_, shrink=0.7)
        a_.set_title(lab)
    return png(fig)


def fig_3d(hag, dgm, surf, r_, nw, H, res):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from types import SimpleNamespace
    r_ = SimpleNamespace(**r_.to_dict())                          # a Series has a method .shift - use plain attributes
    i0, j0 = int(r_.i0), int(r_.j0)
    hw_, dw_, sf_ = hag[i0:i0 + nw, j0:j0 + nw], dgm[i0:i0 + nw, j0:j0 + nw], surf[i0:i0 + nw, j0:j0 + nw]
    jj, ii = np.meshgrid(np.arange(nw), np.arange(nw)); Xw, Yw = (j0 + jj + 0.5) * res, (H - (i0 + ii) - 0.5) * res
    a = np.radians(r_.axis_grid); na, ua = np.array([np.cos(a), -np.sin(a)]), np.array([np.sin(a), np.cos(a)])
    sw = Xw * np.cos(a) - Yw * np.sin(a); lw = Xw * np.sin(a) + Yw * np.cos(a)
    z0 = np.nanmedian(dw_) if np.isfinite(dw_).any() else 0.0
    fig = plt.figure(figsize=(10, 7.5)); ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Xw[::4, ::4], Yw[::4, ::4], np.where(np.isfinite(dw_), dw_, z0)[::4, ::4] - z0, color=(0.6, 0.55, 0.45, 0.5), linewidth=0)
    quads = []
    for t in range(int(np.floor((sw.min() - r_.shift) / r_.pitch)) - 1, int(np.ceil((sw.max() - r_.shift) / r_.pitch)) + 2):
        s_r = r_.shift + t * r_.pitch; s_f = s_r + (r_.flank if r_.side == "right" else -r_.flank)
        for l0 in np.arange(lw.min(), lw.max(), 5.0):
            l1 = min(l0 + 5.0, lw.max()); pts = []
            for s_, h_ in ((s_r, r_.h_ridge), (s_f, r_.h_foot)):
                for l_ in (l0, l1):
                    px, py = s_ * na[0] + l_ * ua[0], s_ * na[1] + l_ * ua[1]
                    if not (0 <= px / res - j0 < nw and 0 <= (H - py / res) - i0 < nw):
                        pts = None; break
                    ci, cj = int(np.clip((H - py / res) - i0, 0, nw - 1)), int(np.clip(px / res - j0, 0, nw - 1))
                    pts.append((px, py, (dw_[ci, cj] if np.isfinite(dw_[ci, cj]) else z0) + h_ - z0))
                if pts is None:
                    break
            if pts:
                quads.append([pts[0], pts[1], pts[3], pts[2]])
    ax.add_collection3d(Poly3DCollection(quads, facecolor=(0.12, 0.27, 0.59, 0.55), edgecolor="k", linewidths=0.3))
    m = np.isfinite(sf_) & (hw_ > GW.MIN_H)
    ax.scatter(Xw[m][::3], Yw[m][::3], (sf_ - z0)[m][::3], s=1.2, c="#B3121A", alpha=0.5)
    ax.set_zlim(-3, 6); ax.set_box_aspect((1, 1, 0.25)); ax.view_init(elev=28, azim=-55)
    ax.set_title(f"3-D-Modell Fenster {int(r_.w)}: Neigung {r_.tilt_horiz:.1f}°, Pitch {r_.pitch:.1f} m, nach {r_.azimuth_true:.0f}°")
    return png(fig)


def cut_line(surf, dgm, r_, nw, H, res):
    """60-m section across the rows through the centre of the example window."""
    if r_ is None:
        return None
    cx, cy = (r_.j0 + nw / 2) * res, (H - r_.i0 - nw / 2) * res
    a = np.radians(r_.axis_grid); na = np.array([np.cos(a), -np.sin(a)])
    s = np.arange(-30, 30, res)
    xs, ys = cx + s * na[0], cy + s * na[1]
    jj = np.clip((xs / res).astype(int), 0, surf.shape[1] - 1); ii = np.clip((H - ys / res).astype(int), 0, surf.shape[0] - 1)
    return dict(x=s + 30, surf=surf[ii, jj], dgm=dgm[ii, jj], line=(xs, ys))


# ------------------------------------------------------------------------------------------------ page
CSS = """body{font-family:system-ui,sans-serif;margin:16px;max-width:1500px;color:#222}h1{font-size:22px}h2{font-size:17px;margin-top:28px;
border-bottom:1px solid #ccc}table.t{border-collapse:collapse;font-size:12px;margin:6px 0}table.t td,table.t th{border:1px solid #ddd;padding:2px 6px;
text-align:right}table.kv th{text-align:left;background:#f4f4f4}.ok{color:#11692b;font-weight:600}.bad{color:#a11;font-weight:600}.note{font-size:12px;color:#555}"""


def page(title: str, parts: list) -> str:
    body = "\n".join(parts)
    return f"<!doctype html><html lang='de'><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{CSS}</style></head><body>{body}</body></html>"


def build_page(P: dict, win: pd.DataFrame, G: pd.DataFrame, arrays: pd.DataFrame, park: dict, head: dict, compare: pd.DataFrame,
               src_rows: pd.DataFrame = None, extra: list = None) -> str:
    """P: rasters and meta (hag, dgm, surf, inside, rgb, conv, RES, H, extent, ortho_rows, source, survey_year, pts, title)."""
    res, H = P["RES"], P["H"]
    nw, win_m = int(round(park["win_m"] / res)), float(park["win_m"])
    acc = win[win.ok.astype(bool)] if len(win) else win
    ex = None
    if len(acc):
        c = (acc.x - acc.x.mean()) ** 2 + (acc.y - acc.y.mean()) ** 2 - 2e4 * acc.f_tab
        ex = int(acc.loc[c.idxmin(), "w"])
    elif len(win) and (win.f_tab >= 0.3).any():
        ex = int(win.loc[win.f_tab.idxmax(), "w"])
    st = park.get("status", "")
    parts = [f"<h1>{html.escape(P['title'])}</h1>",
             f"<p>Parkannahme: <span class='{'ok' if st == 'accepted' else 'bad'}'>{html.escape(st)}</span> {html.escape(str(park.get('status_reason', '')))}"
             f" · Quelle {html.escape(str(P['source']))}, Befliegung {P['survey_year']} · Fenster {park.get('n_ok')}/{park.get('n_windows')} angenommen"
             f" · gelockerte Annahme (≥ 2 Fenster): {park.get('accept_relaxed')}</p>",
             "<h2>1. Stammdaten und Zuordnung</h2>", kv(head)]
    if src_rows is not None and len(src_rows):
        parts += ["<p class='note'>Alle Oberflächenquellen dieses Parks:</p>", table(src_rows)]
    parts += ["<h2>2. Umring und Orthophoto</h2>", fig_outline(P.get("rgb"), P["inside"], P["extent"], None, P.get("pts", []), "DOP20, rot = Analysemaske (Umring − 5 m)")]
    if P.get("rgb") is not None and P.get("ortho_rows") is not None:
        parts += ["<h2>3. Graubild und 2-D-Spektrum</h2>", fig_spectrum(P["rgb"], P["inside"], res, P["conv"], P["ortho_rows"]),
                  "<h2>4. Fensterkarte des Orthophotos</h2>", fig_ortho_windows(P["rgb"], P["extent"], H, res, P["ortho_rows"])]
    r_ex = win[win.w == ex].iloc[0] if ex is not None else None
    parts += [f"<h2>5. Oberfläche ({html.escape(str(P['source']))}, Befliegung {P['survey_year']}) und Höhe über Grund</h2>",
              f"<p class='note'>Befliegungsdaten: {html.escape(str(P.get('survey_dates'))[:300])}</p>",
              fig_surface(P["surf"], P["dgm"], P["hag"], P["inside"], P["extent"], res, P["source"], cut_line(P["surf"], P["dgm"], r_ex, nw, H, res) if r_ex is not None and "axis_grid" in win and pd.notna(r_ex.get("axis_grid")) else None)]
    if len(win):
        parts += ["<h2>6. Fenstergitter</h2>", fig_grid(P["hag"], win, nw, win_m, P["extent"], H, res, ex)]
    if r_ex is not None:
        res_w, det = GW.measure_window(P["hag"], P["dgm"], (int(r_ex.i0), int(r_ex.j0), float(r_ex.f_in), float(r_ex.f_tab)), nw, P["conv"], P["source"], res, w=ex, detail=True)
        if "h_ridge" in res_w:
            parts += ["<h2>7. Beispielfenster ausführlich</h2>", fig_window(det, res_w),
                      "<details><summary>alle Größen des Beispielfensters</summary>" + kv({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res_w.items()}) + "</details>"]
    if len(win):
        parts += ["<h2>8. Karten je Fenster</h2>", f"<p class='note'>Verworfene Fenster mit Tischen schraffiert, Grund als Kürzel: {SHORT_TXT}. "
                  "„?“ am Azimut: Seitenwahl ohne Grundlage.</p>", fig_maps(P["hag"], win, nw, win_m, P["extent"], H, res)]
        rej = win[(~win.ok.astype(bool)) & (win.f_tab >= GW.F_TAB_MIN)]
        if len(rej):
            parts += ["<p class='note'>Verwerfungsgründe (Fenster mit Tischen): " +
                      html.escape(str(rej.reason.value_counts().to_dict())) + "</p>"]
    parts += ["<h2>9. Gruppen und Arrays für den Generator</h2>", table(G.drop(columns=["windows"], errors="ignore")) if len(G) else "<p>keine Gruppe</p>",
              table(arrays) if len(arrays) else "<p>keine Arrays (Registergeometrie)</p>",
              "<p class='note'>Flags: " + html.escape(", ".join(f"{k}={park.get(k)}" for k in ("rej_tab_frac", "warn_rejected", "az_undetermined", "az_conflict", "slope_cover_min", "n_splinter", "box", "box_side_m") if k in park)) + "</p>"]
    if r_ex is not None and "h_ridge" in win and pd.notna(r_ex.get("h_ridge")):
        parts += ["<h2>10. 3-D-Modell des Beispielfensters</h2>", fig_3d(P["hag"], P["dgm"], P["surf"], r_ex, nw, H, res)]
    parts += ["<h2>11. Vergleich mit dem Register</h2>", table(compare)]
    parts += ["<details><summary>Fenstertabelle (alle Variablen)</summary>", table(win.drop(columns=["x", "y"], errors="ignore"), 3), "</details>"]
    if extra:
        parts[1:1] = ["<p><a href='#synthese'>→ Teil B: Synthese</a></p>"]
        parts += extra
    parts += [f"<p class='note'>Erzeugt {pd.Timestamp.now():%Y-%m-%d %H:%M} mit scripts/pv/park_report.py (Fenstermethode pv.geodata_windows). "
              "Orthophotos und Höhenmodelle: GeoBasis-DE / Landesvermessungen der Länder (Lizenzen siehe pv/geodata_states.py).</p>"]
    return page(P["title"], parts)


# ------------------------------------------------------------------------------------------------ fleet
def short_page(pid: str, r: pd.Series, row: dict, out_dir: str) -> None:
    """Page of a park without geometry measurement: register entry, reason, synthesis part."""
    import synthesis_report as SR
    head = {"Park": pid, "Land / Kreis": f"{r.bundesland} / {r.landkreis}", "Leistung DC / AC": f"{r.dc_kwp:.0f} kWp / {r.ac_kw:.0f} kW",
            "Inbetriebnahme": int(r.year), "Einheiten": int(r.n_units), "Registerklasse": f"{r.orientation_class} / {r.tilt_class}",
            "Fläche laut MaStR": f"{r.area_ha} ha", "Betreiber": r.operator, "Koordinate (Register)": f"{r.lat:.5f}, {r.lon:.5f}"}
    parts = [f"<h1>{html.escape(f'{pid} – {r.bundesland}, {r.dc_kwp:.0f} kWp, {r.year}')}</h1>",
             f"<p>Teil A (Geometriemessung): <span class='bad'>keine Messung</span> – {html.escape(str(row.get('note', '')))}</p>",
             "<h2>1. Stammdaten</h2>", kv(head)] + SR.section(pid, start=2)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{pid}.html")
    open(path, "w").write(page(pid, parts))
    row["mb"] = round(os.path.getsize(path) / 1e6, 1)


def fleet_report(pid: str, suffix: str, out_dir: str, synthesis: bool = False) -> dict:
    import geometry_release_states as R
    from parks_pv import paths
    from pyproj import Transformer
    from shapely.geometry import Point
    sel = pd.read_csv(paths.SELECTION, dtype={"park_id": str}).set_index("park_id")
    r = sel.loc[pid].copy()
    r["park_id"] = pid                                            # fetch_margin and municipality use r.park_id
    state = r.bundesland
    row = dict(park_id=pid, state=state, dc_kwp=float(r.dc_kwp), year=int(r.year), orientation_class=r.orientation_class, tilt_class=r.tilt_class)
    csv = os.path.join(paths.DATA_DIR, f"geodata_release_{state.lower()}{suffix}.csv")
    rd = os.path.join(REPO, "results", "pv", f"geodata_release{suffix}", pid)
    if state not in R.STATE or not os.path.exists(csv):
        row.update(status="register", source="", note="kein Abrufer für das Land / kein Lauf")
        if synthesis:
            short_page(pid, r, row, out_dir)
        return row
    df = pd.read_csv(csv, dtype={"pv": str}); dp = df[df.pv == pid]
    best = dp[dp.best] if "best" in dp else dp.iloc[:0]
    pick = best if len(best) else dp.sort_values("source_rank") if len(dp) else dp
    if not len(pick) or pd.isna(pick.source.iloc[0]) or not os.path.exists(os.path.join(rd, f"windows_{pick.source.iloc[0]}.csv")):
        row.update(status="failed", source="", note=str(dp.status_reason.iloc[0]) if len(dp) else "")
        if synthesis:
            short_page(pid, r, row, out_dir)
        return row
    src = pick.source.iloc[0]
    win = read_csv0(os.path.join(rd, f"windows_{src}.csv")); G = read_csv0(os.path.join(rd, f"groups_{src}.csv")); arrays = read_csv0(os.path.join(rd, f"arrays_{src}.csv"))
    park = json.load(open(os.path.join(rd, f"park_{src}.json")))
    code, zone = R.STATE[state]
    osm = json.load(open(os.path.join(paths.INPUT_DIR, "osm", "parks_osm.json")))
    tiles = R.fetch_tiles(state, float(r.lat), float(r.lon), int(r.year), r.landkreis, R.municipality(r.unit_ids) if state == "Hessen" else None,
                          margin_m=R.fetch_margin(r, osm, zone))
    to_m = Transformer.from_crs("EPSG:4326", GT.CRS[zone], always_xy=True).transform
    pt = Point(*to_m(float(r.lon), float(r.lat)))
    if park.get("box"):
        half = float(park["box_side_m"]) / 2
        bounds = (pt.x - half, pt.y - half, pt.x + half, pt.y + half)
        H, W = int(round(2 * half / GW.RES)), int(round(2 * half / GW.RES))
        inside = np.ones((H, W), bool)
    else:
        poly, _, _ = R.footprint(r, osm, to_m, pt)
        g = GW.analysis_grid(poly, GW.RES); bounds, H, inside = g["bounds"], g["H"], g["inside"]
    ras = R.rasters(tiles, src, dict(R.sources_from(tiles))[src], bounds, zone, float(r.lat), float(r.lon))
    rgb = GT.mosaic(tiles["dop20"], bounds, GW.RES, zone, bands=[1, 2, 3]) if tiles.get("dop20") else None
    ortho = GW.orthophoto_axis(rgb, inside, GW.RES)["rows"] if rgb is not None else None
    P = dict(hag=ras["hag"], dgm=ras["dgm"], surf=ras["surf"], inside=inside, rgb=rgb, conv=ras["conv"], RES=GW.RES, H=H,
             extent=[0, inside.shape[1] * GW.RES, 0, H * GW.RES], ortho_rows=ortho, source=src, survey_year=ras["survey_year"],
             survey_dates=ras["survey_dates"], title=f"{pid} – {state}, {r.dc_kwp:.0f} kWp, {r.year}",
             pts=[("Registerkoordinate", (pt.x - bounds[0], pt.y - bounds[1]), "bx")])
    G0 = G.iloc[0] if len(G) else None
    head = {"Park": pid, "Land / Kreis": f"{state} / {r.landkreis}", "Leistung DC / AC": f"{r.dc_kwp:.0f} kWp / {r.ac_kw:.0f} kW",
            "Inbetriebnahme": int(r.year), "Einheiten": int(r.n_units), "Registerklasse": f"{r.orientation_class} / {r.tilt_class}",
            "Fläche laut MaStR": f"{r.area_ha} ha", "Betreiber": r.operator, "Umring": "Kasten" if park.get("box") else "OSM",
            "Koordinate (Register)": f"{r.lat:.5f}, {r.lon:.5f}"}
    compare = pd.DataFrame([dict(Größe="Orientierung", Register=r.orientation_class, gemessen=pick.or_cls_measured.iloc[0], stimmt=pick.or_agree.iloc[0]),
                            dict(Größe="Neigungsklasse", Register=r.tilt_class, gemessen=pick.tilt_cls_measured.iloc[0], stimmt=pick.tilt_agree.iloc[0]),
                            dict(Größe="Neigung (°)", Register=r.tilt_class, gemessen=round(float(pick.tilt_measured.iloc[0]), 1) if pd.notna(pick.tilt_measured.iloc[0]) else None,
                                 stimmt=pick.tilt_agree_tol.iloc[0])])
    src_rows = dp[["source", "status", "n_ok", "n_windows", "tab_share_accepted", "g0_tilt", "g0_azimuth", "g0_ew", "survey_year", "best"]].copy()
    extra = None
    if synthesis:
        import synthesis_report as SR
        extra = SR.section(pid, start=12)
    html_ = build_page(P, win, G, arrays, park, head, compare, src_rows, extra)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{pid}.html")
    open(path, "w").write(html_)
    row.update(status=park["status"], source=src, survey=ras["survey_year"], n_ok=park["n_ok"], n_windows=park["n_windows"],
               tilt=round(float(G0.tilt_horiz), 1) if G0 is not None else None, azimuth=round(float(arrays.azimuth.iloc[0])) if len(arrays) else None,
               ew=bool(G0.ew) if G0 is not None else False, or_agree=pick.or_agree.iloc[0], tilt_agree_tol=pick.tilt_agree_tol.iloc[0],
               accept_relaxed=park.get("accept_relaxed"), footprint="Kasten" if park.get("box") else "OSM", mb=round(os.path.getsize(path) / 1e6, 1))
    return row


def trianel_report(pv: str, out_dir: str) -> dict:
    import geometry_trianel_windows as TW
    P0 = TW.prepare(pv)
    d = TW.OUTD
    win = pd.read_csv(os.path.join(d, f"{pv}_windows.csv")); G = pd.read_csv(os.path.join(d, f"{pv}_groups.csv")); arrays = pd.read_csv(os.path.join(d, f"{pv}_arrays.csv"))
    park = json.load(open(os.path.join(d, f"{pv}_park.json")))
    ortho = P0["ortho_rows"]
    pts = [("Trianel", (P0["pt_tri"].x - P0["minx"], P0["pt_tri"].y - P0["miny"]), "r+")]
    if P0["pt_reg"] is not None:
        pts.append(("Register", (P0["pt_reg"].x - P0["minx"], P0["pt_reg"].y - P0["miny"]), "bx"))
    P = dict(hag=P0["hag"], dgm=P0["dgm"], surf=P0["surf"], inside=P0["inside"], rgb=P0["rgb"], conv=P0["conv"], RES=P0["RES"], H=P0["H"],
             extent=P0["extent"], ortho_rows=ortho, source=P0["source"], survey_year=P0["survey_year"], survey_dates=P0["survey_dates"],
             title=f"{pv} (Trianel, vertraulich) – {P0['state']}, {P0['kw_master']:.0f} kW", pts=pts)
    head = {"Park": pv, "Land": P0["state"], "Leistung (Stammdaten)": f"{P0['kw_master']:.0f} kW", "Inbetriebnahme (Register)": P0["year_built"],
            "Registerklasse": f"{P0['register_orientation']} / {P0['register_tilt_class']}", "Umring": P0["fp_src"],
            "Betreiberwinkel (tilt0, tilt1, az0, az1, kW)": str([(float(x.tilt0), float(x.tilt1), float(x.az0), float(x.az1), float(x.kw)) for x in P0["rows"].itertuples()])}
    G0 = G.iloc[0] if len(G) else None
    compare = pd.DataFrame([dict(Größe="Neigung (°)", Betreiber=float(P0["rows"].tilt0.iloc[0]), gemessen=round(float(G0.tilt_horiz), 1) if G0 is not None else None),
                            dict(Größe="Azimut (°)", Betreiber=float(P0["rows"].az0.iloc[0]), gemessen=round(float(arrays.azimuth.iloc[0])) if len(arrays) else None)])
    html_ = build_page(P, win, G, arrays, park, head, compare)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{pv}.html")
    open(path, "w").write(html_)
    return dict(park_id=pv, state=P0["state"], status=park["status"], source=P0["source"], survey=P0["survey_year"], n_ok=park["n_ok"],
                n_windows=park["n_windows"], tilt=round(float(G0.tilt_horiz), 1) if G0 is not None else None,
                azimuth=round(float(arrays.azimuth.iloc[0])) if len(arrays) else None, ew=bool(G0.ew) if G0 is not None else False,
                mb=round(os.path.getsize(path) / 1e6, 1))


INDEX_JS = """<script>
function f(){const s=document.getElementById('fs').value,q=document.getElementById('fq').value,a=document.getElementById('fa').value;
document.querySelectorAll('#ix tbody tr').forEach(r=>{const c=r.dataset;r.style.display=((!s||c.state==s)&&(!q||c.source==q)&&(!a||c.status==a))?'':'none'})}
</script>"""


def index(rows: list, out_dir: str, title: str) -> None:
    df = pd.DataFrame(rows)
    opts = lambda col: "".join(f"<option>{html.escape(str(v))}</option>" for v in sorted(df[col].dropna().astype(str).unique())) if col in df else ""
    cols = [c for c in ["park_id", "state", "dc_kwp", "year", "orientation_class", "tilt_class", "status", "source", "survey", "footprint", "n_ok", "n_windows",
                        "tilt", "azimuth", "ew", "or_agree", "tilt_agree_tol", "accept_relaxed", "mb", "note"] if c in df]
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = ""
    for r in df.itertuples():
        link = f"<a href='{r.park_id}.html'>{r.park_id}</a>" if os.path.exists(os.path.join(out_dir, f"{r.park_id}.html")) else r.park_id
        cells = "".join(f"<td>{link if c == 'park_id' else html.escape('' if pd.isna(getattr(r, c)) else str(getattr(r, c)))}</td>" for c in cols)
        body += f"<tr data-state='{html.escape(str(getattr(r, 'state', '')))}' data-source='{html.escape(str(getattr(r, 'source', '')))}' data-status='{html.escape(str(getattr(r, 'status', '')))}'>{cells}</tr>"
    filt = (f"Land <select id='fs' onchange='f()'><option value=''>alle</option>{opts('state')}</select> "
            f"Quelle <select id='fq' onchange='f()'><option value=''>alle</option>{opts('source')}</select> "
            f"Annahme <select id='fa' onchange='f()'><option value=''>alle</option>{opts('status')}</select>")
    summ = df.status.value_counts().to_dict() if "status" in df else {}
    html_ = page(title, [f"<h1>{html.escape(title)}</h1>", f"<p>{len(df)} Parks; Status {html.escape(str(summ))}</p>", filt,
                         f"<table class='t' id='ix'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>", INDEX_JS])
    open(os.path.join(out_dir, "index.html"), "w").write(html_)
    df.to_csv(os.path.join(out_dir, "index.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fleet", "trianel"])
    ap.add_argument("--parks", default=None)
    ap.add_argument("--pv", action="append", default=None)
    ap.add_argument("--suffix", default="_v2")
    ap.add_argument("--out", default=None)
    ap.add_argument("--synthesis", action="store_true", help="append the synthesis part (scripts/pv/synthesis_report.py) of $PV_SYNTH_RELEASE (default parks_pv_v2)")
    a = ap.parse_args()
    rows = []
    if a.mode == "fleet":
        from parks_pv import paths
        out = a.out or OUT_FLEET
        sel = pd.read_csv(paths.SELECTION, dtype={"park_id": str})
        ids = a.parks.split(",") if a.parks else sel.park_id.tolist()
        for pid in ids:
            try:
                rows.append(fleet_report(pid, a.suffix, out, a.synthesis))
            except Exception as ex:
                import traceback
                traceback.print_exc()
                r = sel.set_index("park_id").loc[pid]
                rows.append(dict(park_id=pid, state=r.bundesland, status="report_error", note=f"{type(ex).__name__}: {str(ex)[:120]}"))
            print(rows[-1], flush=True)
        meta = sel.set_index("park_id")[["dc_kwp", "year", "orientation_class", "tilt_class"]]
        rows = [dict(r, **{k: meta.loc[r["park_id"], k] for k in meta.columns if k not in r}) for r in rows]
        if a.parks and os.path.exists(os.path.join(out, "index.csv")):   # update the rows of these parks in the existing index
            old = pd.read_csv(os.path.join(out, "index.csv"), dtype={"park_id": str})
            new = pd.DataFrame(rows)
            rows = pd.concat([old[~old.park_id.isin(new.park_id)], new]).set_index("park_id").reindex(sel.park_id).reset_index().to_dict("records")
            a.parks = None
        if not a.parks:
            index(rows, out, "Geometrie der Flotte (Fenstermethode, v2)")
    else:
        out = a.out or OUT_TRIANEL
        for pv in a.pv or ["pv1", "pv2", "pv3", "pv5", "pv6", "pv7"]:
            rows.append(trianel_report(pv, out)); print(rows[-1], flush=True)
        index(rows, out, "Trianel-Parks (vertraulich, nur l2)")


if __name__ == "__main__":
    main()
