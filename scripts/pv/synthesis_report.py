#!/usr/bin/env python
"""Synthesis part of the park report (2026-10-23): what the generator made of one park of a release.

Sections (HTML parts appended to the geometry page of park_report.py, or a page of their own for parks without a
geometry measurement):
  chain and provenance (manifest, drivers, reproduction of power_park from the manifest config);
  sub-arrays used by the generator next to v1 and the register view;
  park parameters (power, modules, aging, losses, area/GCR, terrain, horizon with the sun paths);
  loss chain per kWp (GHI -> POA without shading -> horizon -> inter-row -> IAM -> DC -> aging -> DC losses ->
  inverter -> clipping -> AC losses/availability -> snow), recomputed with the chain switched stage by stage;
  yield (monthly energy v2 vs v1, ratio, specific yield per year, hour x day map);
  example weeks (clear summer week, snow week);
  snow model (depth_linear: coverage = clip(ERA5 depth / 5 cm, 0, 1), factor = 1 - dc_loss_nrel(coverage, 100)),
  per winter and in the snowiest week; drivers (CAMS raw vs corrected, ERA5).

    synthre/bin/python scripts/pv/park_report.py fleet --parks ID --synthesis --out DIR
Inputs: release dir ($PV_SYNTH_RELEASE under synthetic/solar, default parks_pv_v2; independent of PV_RELEASE_NAME,
which selects the geometry tables), its work/chain drivers,
inputs/horizons_used.json, parks_pv_v1 for the comparison. Public register parks only.
"""
import copy
import html
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

import park_report as PR  # noqa: E402  (png, table, kv)
from parks_pv import generate as GEN, paths  # noqa: E402

ROOT_SOLAR = os.path.join(paths.ROOT, "synthetic", "solar")
V1_DIR = os.path.join(ROOT_SOLAR, "parks_pv_v1")
C_V2, C_V1, C_AQUA, C_YEL, C_GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"
OR_AZ = {"Nord": 0, "Nord-Ost": 45, "Ost": 90, "Süd-Ost": 135, "Süd": 180, "Süd-West": 225, "West": 270, "Nord-West": 315}
TILT_CLS = {"< 20 Grad": (0, 20), "20 - 40 Grad": (20, 40), "40 - 60 Grad": (40, 60), "> 60 Grad": (60, 90)}


# ------------------------------------------------------------------------------------------------ data
def load(pid: str, rel_dir: str = None) -> dict:
    """All release data of one park: manifest, park/site rows, arrays (v2, v1), register view, frames, drivers, horizon."""
    rel = rel_dir or os.path.join(ROOT_SOLAR, os.environ.get("PV_SYNTH_RELEASE", "parks_pv_v2"))
    S = dict(pid=pid, rel=rel, name=os.path.basename(rel))
    S["man"] = json.load(open(os.path.join(rel, f"manifest_{pid}.json")))
    S["park"] = pd.read_csv(os.path.join(rel, "parks.csv"), dtype={"park_id": str}).set_index("park_id").loc[pid]
    S["site"] = pd.read_csv(os.path.join(rel, "sites.csv"), dtype={"site_id": str}).set_index("site_id").loc[pid]
    S["arr"] = pd.read_csv(os.path.join(rel, "pv_arrays.csv"), dtype={"park_id": str}).query("park_id == @pid")
    S["view"] = pd.read_csv(os.path.join(rel, "pv_masterdata_view.csv"), dtype={"park_id": str}).set_index("park_id").loc[pid]
    S["df"] = pd.read_parquet(os.path.join(rel, f"synth_{pid}.parquet"))
    S["drv"] = pd.read_parquet(os.path.join(rel, "work", "chain", f"{pid}_drivers.parquet"))
    hz = json.load(open(os.path.join(paths.INPUT_DIR, "horizons_used.json"))).get(pid)
    S["hz"] = pd.Series({float(k): float(v) for k, v in hz.items()}).sort_index() if hz else None
    v1 = os.path.join(V1_DIR, f"synth_{pid}.parquet")
    S["v1"] = pd.read_parquet(v1, columns=["power_park"]).power_park if os.path.exists(v1) else None
    a1 = pd.read_csv(os.path.join(V1_DIR, "pv_arrays.csv"), dtype={"park_id": str})
    S["arr_v1"] = a1[a1.park_id == pid]
    p1 = pd.read_csv(os.path.join(V1_DIR, "parks.csv"), dtype={"park_id": str}).set_index("park_id")
    S["dev_v1"] = p1.register_deviation.get(pid) if pid in p1.index else None
    S["kwp"] = float(S["man"]["params"]["kwp_dc"])
    S["kw_ac"] = float(S["man"]["params"]["ac_kw"])
    return S


def full_years(idx: pd.DatetimeIndex) -> list:
    """Calendar years completely inside the index (UTC)."""
    return [y for y in sorted(set(idx.year)) if idx.min() <= pd.Timestamp(f"{y}-01-01", tz="UTC") and idx.max() >= pd.Timestamp(f"{y}-12-31 23:00", tz="UTC")]


def inverter_unclipped(p_dc: pd.Series, pac0_w: float, eta_nom: float, eta_ref: float) -> pd.Series:
    """pvlib PVWatts inverter efficiency curve without the AC cap (chain passes pdc0 = P_AC, cap = eta_nom * P_AC)."""
    z = (p_dc / pac0_w).where(p_dc > 0)
    eta = eta_nom / eta_ref * (-0.0162 * z - 0.0059 / z + 0.9858)
    return (eta * p_dc).fillna(0.0).clip(lower=0.0)


def loss_chain(S: dict) -> tuple:
    """Rerun the chain from the manifest config: full (reproduction check), without horizon/inter-row, horizon only.
    Returns (stage table per kWp and year, max abs deviation of the rerun from the release in W)."""
    cfg = S["man"]["config"]
    drv = S["drv"]
    full = GEN.run(copy.deepcopy(cfg), drv, horizon=S["hz"])
    dev = float((full["power_park"] - S["df"]["power_park"].reindex(full.index)).abs().max())
    c0 = copy.deepcopy(cfg); c0["shading"] = {"horizon": False, "inter_row": False}
    c1 = copy.deepcopy(cfg); c1["shading"] = {"horizon": True, "inter_row": False}
    r0 = GEN.run(c0, drv, horizon=S["hz"])
    r1 = GEN.run(c1, drv, horizon=S["hz"]) if S["hz"] is not None else r0
    yrs = full_years(full.index)
    sel = full.index.year.isin(yrs)
    n = len(yrs)
    kwp, pac0 = S["kwp"], S["kw_ac"] * 1e3
    iv = cfg["inverter"]
    ac_u = inverter_unclipped(full["p_dc_w"], pac0, float(iv["eta_inv_nom"]), float(iv.get("eta_inv_ref", 0.9637)))
    ac_c = np.minimum(ac_u, float(iv["eta_inv_nom"]) * pac0)
    E = lambda s: float(s[sel].sum()) / 1e3 / n                                  # Wh/h steps -> kWh per year (hourly frames)
    step_h = (full.index[1] - full.index[0]) / pd.Timedelta("1h")
    rows = [("GHI (korrigiertes CAMS)", E(full["ghi"]) * step_h, "kWh/m²"),
            ("POA global ohne Verschattung", E(r0["poa_global"]) * step_h, "kWh/m²"),
            ("POA nach Horizont", E(r1["poa_global"]) * step_h, "kWh/m²"),
            ("POA nach Reihenverschattung", E(full["poa_global"]) * step_h, "kWh/m²"),
            ("POA effektiv (IAM)", E(full["poa_eff"]) * step_h, "kWh/m²"),
            ("DC (Temperatur, γ)", E(full["p_dc_raw_w"]) * step_h / kwp, "kWh/kWp"),
            ("DC nach Alterung", E(full["p_dc_raw_w"] * full["aging_factor"]) * step_h / kwp, "kWh/kWp"),
            ("DC nach DC-Verlusten", E(full["p_dc_w"]) * step_h / kwp, "kWh/kWp"),
            ("AC nach Wechselrichter (ohne Deckel)", E(ac_u) * step_h / kwp, "kWh/kWp"),
            ("AC nach Clipping", E(ac_c) * step_h / kwp, "kWh/kWp"),
            ("AC nach AC-Verlusten und Verfügbarkeit", E(full["p_ac_w"]) * step_h / kwp, "kWh/kWp"),
            ("AC nach Schnee (power_park)", E(full["power_park"]) * step_h / kwp, "kWh/kWp")]
    t = pd.DataFrame(rows, columns=["Stufe", "Wert", "Einheit"])
    t["Änderung %"] = (t.Wert / t.Wert.shift(1) - 1) * 100
    t.loc[t.Einheit != t.Einheit.shift(1), "Änderung %"] = np.nan                # unit change POA -> DC: no ratio
    t["PR-Bezug"] = np.where(t.Einheit == "kWh/kWp", t.Wert / t.loc[3, "Wert"], np.nan)  # performance ratio vs shaded POA
    return t, dev, yrs, full


# ------------------------------------------------------------------------------------------------ figures
def fig_arrays(S: dict) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    v = S["view"]
    if v.orientation_class in OR_AZ and v.tilt_class in TILT_CLS:
        a0, (t0, t1) = OR_AZ[v.orientation_class], TILT_CLS[v.tilt_class]
        ax.add_patch(plt.Rectangle((a0 - 22.5, t0), 45, t1 - t0, color=C_GREY, alpha=0.18, lw=0))
        ax.text(0.02, 0.96, f"grau: Registerklasse {v.orientation_class} / {v.tilt_class}", transform=ax.transAxes, fontsize=8, color="#52514e", va="top")
    if len(S["arr"]):
        a = S["arr"]
        ax.scatter(a.azimuth, a.tilt, s=30 + 300 * a.share, color=C_V2, alpha=0.85, edgecolor="white", linewidth=1.5, label=f"{S['name']} (Fläche ∝ Anteil)", zorder=3)
    if len(S["arr_v1"]):                                          # hollow on top: identical arrays stay visible
        a = S["arr_v1"]
        ax.scatter(a.azimuth, a.tilt, s=30 + 300 * a.share, facecolor="none", edgecolor=C_V1, linewidth=2, label="v1 (Ring)", zorder=4)
    az = pd.concat([S["arr"].azimuth, S["arr_v1"].azimuth])
    ax.set_xlim(min(60, az.min() - 15), max(300, az.max() + 15)); ax.set_ylim(0, max(48, pd.concat([S["arr"].tilt, S["arr_v1"].tilt]).max() + 5))
    ax.set_xlabel("Azimut (°, 180 = Süd)"); ax.set_ylabel("Neigung (°)")
    ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="lower right", frameon=False)
    return PR.png(fig)


def fig_horizon(S: dict) -> str:
    import pvlib
    lat, lon = float(S["park"].latitude), float(S["park"].longitude)
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    for day, c in (("2025-06-21", C_YEL), ("2025-03-20", C_AQUA), ("2025-12-21", C_V2)):
        t = pd.date_range(f"{day} 00:00", f"{day} 23:55", freq="5min", tz="UTC")
        sp = pvlib.solarposition.get_solarposition(t, lat, lon)
        sp = sp[sp.apparent_elevation > -2]
        ax.plot(sp.azimuth, sp.apparent_elevation, color=c, lw=1.5, label=f"Sonnenbahn {day[5:]}")
    if S["hz"] is not None:
        ax.fill_between(S["hz"].index, 0, S["hz"].values, color="#52514e", alpha=0.45, step="mid", label=f"Horizont ({S['site'].horizon_src})")
    ax.set_xlim(0, 360); ax.set_ylim(0, 70); ax.set_xticks(range(0, 361, 45))
    ax.set_xlabel("Azimut (°)"); ax.set_ylabel("Höhe (°)"); ax.grid(alpha=0.25); ax.legend(fontsize=7, ncol=4, loc="upper left", frameon=False)
    return PR.png(fig)


def fig_losses(t: pd.DataFrame) -> str:
    d = t.dropna(subset=["Änderung %"]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.barh(d.Stufe, d["Änderung %"], color=[C_V2 if x >= 0 else C_V1 for x in d["Änderung %"]], height=0.6)
    for y, x in enumerate(d["Änderung %"]):
        ax.text(x + (0.3 if x >= 0 else -0.3), y, f"{x:+.1f} %", va="center", ha="left" if x >= 0 else "right", fontsize=8)
    ax.axvline(0, color="#52514e", lw=0.8)
    lo, hi = d["Änderung %"].min(), d["Änderung %"].max()
    ax.set_xlim(min(lo * 1.6, -2), max(hi * 1.35, 2)); ax.set_xlabel("Änderung gegenüber der Vorstufe (%)")
    ax.grid(axis="x", alpha=0.25)
    return PR.png(fig)


def fig_monthly(S: dict) -> str:
    m2 = S["df"].power_park.resample("MS").sum() / 1e6 * S["step_h"]
    fig, axs = plt.subplots(2, 1, figsize=(8.5, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    axs[0].plot(m2.index, m2.values, color=C_V2, lw=2, marker="o", ms=3, label=S["name"])
    if S["v1"] is not None:
        m1 = S["v1"].resample("MS").sum() / 1e6 * S["step_h"]
        axs[0].plot(m1.index, m1.values, color=C_V1, lw=2, marker="o", ms=3, label="parks_pv_v1")
        r = (m2 / m1).where(m1 > 0.02 * m1.max())
        axs[1].plot(r.index, r.values, color="#52514e", lw=1.5, marker="o", ms=3)
        axs[1].axhline(1, color=C_GREY, lw=0.8)
        axs[1].set_ylabel(f"{S['name'][-2:]} / v1")
    else:
        axs[1].text(0.5, 0.5, "kein v1 (Park in v2 ersetzt)", transform=axs[1].transAxes, ha="center")
    axs[0].set_ylabel("Energie (MWh/Monat)"); axs[0].legend(fontsize=8, frameon=False); axs[0].set_title("Monatsenergie (Monat beginnt UTC)", fontsize=9)
    for a in axs:
        a.grid(alpha=0.25)
    return PR.png(fig)


def fig_heatmap(S: dict, year: int) -> str:
    p = S["df"].power_park.loc[str(year)] / (S["kw_ac"] * 1e3)
    m = p.groupby([p.index.hour, p.index.dayofyear]).mean().unstack()
    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    im = ax.imshow(m.values, aspect="auto", origin="lower", cmap="Blues", vmin=0, vmax=1, extent=[1, m.shape[1] + 1, -0.5, 23.5], interpolation="nearest")
    ax.set_xlabel(f"Tag im Jahr {year}"); ax.set_ylabel("Stunde (UTC)"); ax.set_yticks([0, 6, 12, 18, 23])
    fig.colorbar(im, ax=ax, label="P / P_AC", pad=0.01)
    return PR.png(fig)


def fig_week(S: dict, start: pd.Timestamp, title: str, snow: bool = False) -> str:
    w = S["df"].loc[start:start + pd.Timedelta("7D") - pd.Timedelta("1s")]
    n = 3 if snow else 2
    fig, axs = plt.subplots(n, 1, figsize=(8.5, 1.9 * n + 0.6), sharex=True)
    axs[0].plot(w.index, w.ghi, color=C_YEL, lw=1.5, label="GHI")
    axs[0].plot(w.index, w.poa_global, color=C_V2, lw=1.5, label="POA (Arrays gewichtet)")
    axs[0].set_ylabel("W/m²"); axs[0].legend(fontsize=7, ncol=2, frameon=False, loc="upper right"); axs[0].set_title(title, fontsize=9)
    k = 1
    if snow:
        axs[1].plot(w.index, w.snow_depth_cm, color=C_GREY, lw=1.5, label="ERA5-Schneehöhe (cm)")
        axs[1].plot(w.index, w.snow_coverage * 5, color=C_AQUA, lw=1.5, label="Bedeckung × 5")
        axs[1].set_ylabel("cm"); axs[1].legend(fontsize=7, ncol=2, frameon=False, loc="upper right")
        k = 2
    ac = S["kw_ac"] / 1e3
    axs[k].plot(w.index, w.power_park_free / 1e6, color=C_V1, lw=1.2, ls="--", label="ohne Schnee")
    axs[k].plot(w.index, w.power_park / 1e6, color=C_V2, lw=1.8, label="power_park")
    axs[k].axhline(ac, color="#52514e", lw=0.8, ls=":"); axs[k].text(w.index[0], ac, " P_AC", va="bottom", fontsize=7, color="#52514e")
    axs[k].set_ylabel("MW"); axs[k].set_ylim(0, ac * 1.12); axs[k].legend(fontsize=7, ncol=2, frameon=False, loc="upper right")
    for a in axs:
        a.grid(alpha=0.25)
    fig.autofmt_xdate()
    return PR.png(fig)


def fig_snow_curve() -> str:
    from pv import snow
    d = np.linspace(0, 10, 101)
    cov = pd.Series(np.clip(d / GEN.SNOW_FULL_COVER_CM, 0, 1))
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    ax.plot(d, snow.loss_factor(cov).values, color=C_V2, lw=2)
    ax.set_xlabel("ERA5-Schneehöhe (cm)"); ax.set_ylabel("Leistungsfaktor"); ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.25)
    ax.set_title("depth_linear, 100 Strings", fontsize=9)
    return PR.png(fig)


def fig_snow_winters(S: dict) -> str:
    df = S["df"]
    dd = df.snow_depth_cm.resample("D").max()
    lost = ((df.power_park_free - df.power_park).resample("D").sum() / (df.power_park_free.resample("D").sum().replace(0, np.nan))).fillna(0) * 100
    fig, axs = plt.subplots(2, 1, figsize=(8.5, 3.8), sharex=True)
    axs[0].fill_between(dd.index, 0, dd.values, color=C_GREY, alpha=0.6, step="mid"); axs[0].set_ylabel("Schneehöhe\nmax/Tag (cm)")
    axs[1].fill_between(lost.index, 0, lost.values, color=C_V1, alpha=0.8, step="mid"); axs[1].set_ylabel("Schneeverlust\n(% der Tagesenergie)")
    axs[1].set_ylim(0, 105)
    for a in axs:
        a.grid(alpha=0.25)
    return PR.png(fig)


# ------------------------------------------------------------------------------------------------ section
def winters(df: pd.DataFrame, step_h: float) -> pd.DataFrame:
    rows = []
    for y in range(df.index.min().year, df.index.max().year):
        w = df.loc[f"{y}-10-01":f"{y + 1}-04-30"]
        if not len(w):
            continue
        d = w.snow_depth_cm.resample("D").max()
        lost = float((w.power_park_free - w.power_park).sum()) * step_h / 1e6
        rows.append({"Winter": f"{y}/{str(y + 1)[2:]}", "Tage Schnee > 0": int((d > 0.05).sum()), "Tage ≥ 5 cm": int((d >= 5).sum()),
                     "max. Höhe (cm)": round(float(d.max()), 1), "Verlust (MWh)": round(lost, 1),
                     "Verlust (% Okt–Apr)": round(100 * lost / max(float(w.power_park_free.sum()) * step_h / 1e6, 1e-9), 2)})
    return pd.DataFrame(rows)


def section(pid: str, start: int = 12, rel_dir: str = None) -> list:
    """HTML parts of the synthesis part, numbered from start."""
    S = load(pid, rel_dir)
    df, man, par, site, park = S["df"], S["man"], S["man"]["params"], S["site"], S["park"]
    S["step_h"] = (df.index[1] - df.index[0]) / pd.Timedelta("1h")
    t, dev, yrs, full = loss_chain(S)
    cfg = man["config"]
    n = start
    parts = [f"<h1 id='synthese'>Teil B: Synthese ({html.escape(S['name'])})</h1>"]

    # chain and provenance
    drv = man["drivers"]
    prov = {"Release": f"{S['name']}, erzeugt {man.get('created', '')[:16]}, Commit {man.get('git', {}).get('commit', '')[:8]} ({man.get('git', {}).get('branch', '')})",
            "Kette": man["chain"], "Zeitraum": f"{man['period'][0][:16]} → {man['period'][1][:16]}, {man['n_steps']} Schritte à {man['step']} ({man['time_label']})",
            "Strahlung": f"CAMS am Park + 8 Ringpunkte, korrigiert mit cams_bc {drv.get('model_tag')}; CAMS-Rückfall {100 * drv.get('share_hours_cams_fallback', 0):.1f} % der Stunden, Ring fehlt {100 * drv.get('share_hours_ring_missing', 0):.1f} %",
            "ERA5 (Temperatur, Wind, Albedo)": f"Zelle {drv.get('era5_met_cell')} in {drv.get('era5_met_cell_dist_km', np.nan):.1f} km",
            "ERA5 (Strahlung, Vergleich)": f"Zelle {drv.get('era5_solar_cell')} in {drv.get('era5_solar_cell_dist_km', np.nan):.1f} km",
            "Schnee": "ERA5 snow depth, nächste Zelle, Modell depth_linear",
            "Reproduktion": f"power_park aus der Manifest-Konfiguration neu gerechnet: max. Abweichung {dev:.3g} W "
                            + ("<span class='ok'>(identisch)</span>" if dev < 1 else "<span class='bad'>(weicht ab)</span>")}
    parts += [f"<h2>{n}. Kette und Herkunft</h2>", PR.kv({k: v for k, v in prov.items() if k != "Reproduktion"}),
              f"<p>{prov['Reproduktion']}</p>"]
    n += 1

    # arrays
    cols = ["array_id", "tilt", "azimuth", "share", "capacity_kwp", "gcr", "mount_tilt", "mount_azimuth", "geometry_source", "az_source"]
    a2 = S["arr"][[c for c in cols if c in S["arr"]]]
    a1 = S["arr_v1"][[c for c in ["array_id", "tilt", "azimuth", "share", "capacity_kwp", "gcr", "mount_tilt", "mount_azimuth"] if c in S["arr_v1"]]]
    v = S["view"]
    gsrc = str(site.geometry_source)
    expl = {"windows": "gemessen mit der Fenstermethode (Teil A); Neigung gegen die Horizontale, wahrer Azimut, Anteil = Tischfläche",
            "register_v1+dgm1": "Nennwinkel aus v1 (Registerklasse + simulierte Abweichung) auf dem Landes-DGM1 über dem v1-Umring (Hangneigung addiert)",
            "register_v1": "v1 unverändert: Nennwinkel aus v1 auf GLO-30"}
    key = next((k for k in expl if gsrc.startswith(k)), "")
    parts += [f"<h2>{n}. Arrays, die der Generator verwendet</h2>",
              f"<p>Geometriequelle <b>{html.escape(gsrc)}</b>: {html.escape(expl.get(key, ''))}. "
              f"Register: {html.escape(str(v.orientation_class))} / {html.escape(str(v.tilt_class))}, gerundet {v.tilt_deg_rounded5}°"
              f"{', Zweitausrichtung ' + html.escape(str(v.secondary_orientation)) if pd.notna(v.secondary_orientation) else ''}. "
              f"Abweichung zum Register: {html.escape(str(park.register_deviation)) if pd.notna(park.register_deviation) else 'keine'}"
              f" (v1, simuliert: {html.escape(str(S['dev_v1'])) if S['dev_v1'] is not None and pd.notna(S['dev_v1']) else 'keine'}).</p>",
              f"<p class='note'>{html.escape(S['name'])}:</p>", PR.table(a2),
              "<p class='note'>parks_pv_v1:</p>", PR.table(a1) if len(a1) else "<p>kein v1 (Park in v2 ersetzt)</p>",
              fig_arrays(S),
              "<p class='note'>mount_tilt/mount_azimuth = Gestellwinkel; tilt/azimuth = wirksame Winkel inkl. Gelände. GCR aus kWp/ha (Reihenverschattung).</p>"]
    n += 1

    # park parameters
    pp = {"Leistung": f"{S['kwp']:.1f} kWp DC / {S['kw_ac']:.1f} kW AC, DC/AC {par['dc_ac']:.3f}, Leistungsbegrenzung laut MaStR: {par.get('power_limit')}",
          "Module": f"Ära {par['era']}, γ = {par['gamma_pdc']} 1/K (PVWatts), Inbetriebnahme {cfg['aging']['commissioning_date']}, Alterung {'an' if cfg['aging']['enabled'] else 'aus'}",
          "Wechselrichter": f"PVWatts, η_nom {cfg['inverter']['eta_inv_nom']}, η_ref {cfg['inverter'].get('eta_inv_ref')}; AC-Deckel η_nom · P_AC",
          "Verluste": f"DC {cfg['losses']['dc']:.0%}, AC {cfg['losses']['ac']:.0%}, Verfügbarkeit {cfg['losses']['availability']:.0%}, Verschmutzung {'an' if cfg['soiling']['enabled'] else 'aus'}",
          "Temperatur": f"{cfg['temperature']['model']}, u0 = {cfg['temperature']['u0']}, u1 = {cfg['temperature']['u1']}",
          "Einstrahlung": f"{cfg['transposition']['model']}, IAM {cfg['transposition']['iam']}, Albedo {cfg['transposition']['albedo']}; Horizont {'an' if cfg['shading']['horizon'] else 'aus'}, Reihenverschattung {'an' if cfg['shading']['inter_row'] else 'aus'}",
          "Fläche": f"{par['kwp_per_ha']:.0f} kWp/ha ({par['kwp_per_ha_src']}), GCR {par['gcr']:.3f}; Umring {site.footprint_src}, {site.footprint_ha} ha",
          "Gelände": f"{site.altitude:.0f} m ü. NN, Klasse {site.terrain_cls}, Hangneigung im Mittel {site.slope_mean:.1f}° (p90 {site.slope_p90:.1f}°)",
          "Horizont": f"{site.horizon_src}, max. {site.horizon_max_deg}°, Süd {site.horizon_south_deg}°",
          "Archetyp / Client": f"{park.archetype} / {park.client_id} ({park.client_kind}), Größe {park.size_bin}, Kohorte {park.cohort}"}
    if pd.Timestamp(cfg["aging"]["commissioning_date"], tz="UTC") > df.index[0]:
        pp["Vor Inbetriebnahme"] = (f"Die Reihe deckt den ganzen Zeitraum ab; vor {cfg['aging']['commissioning_date']} rechnet der Generator den Park "
                                    "wie neu (Alterungsfaktor 1), es gibt keine Lücke vor der Inbetriebnahme.")
    parts += [f"<h2>{n}. Parkparameter</h2>", PR.kv(pp), fig_horizon(S)]
    n += 1

    # loss chain
    show = t.copy()
    show["Wert"] = show.Wert.round(1); show["Änderung %"] = show["Änderung %"].round(2); show["PR-Bezug"] = show["PR-Bezug"].round(3)
    parts += [f"<h2>{n}. Verlustkette</h2>",
              f"<p class='note'>Mittel der vollen Kalenderjahre {', '.join(map(str, yrs))} (UTC). POA-Stufen durch Neurechnen der Kette mit "
              "abgeschalteter Horizont- bzw. Reihenverschattung; Wechselrichterstufen aus p_dc_w mit der PVWatts-Kennlinie. "
              "PR-Bezug = Energie je kWp / POA nach Reihenverschattung.</p>",
              PR.table(show.fillna("")), fig_losses(t)]
    n += 1

    # yield
    y = []
    for yr in yrs:
        e2 = float(df.power_park.loc[str(yr)].sum()) * S["step_h"] / 1e3
        row = {"Jahr": yr, "Energie (MWh)": round(e2 / 1e3, 1), "kWh/kWp": round(e2 / S["kwp"]), "Volllaststunden AC": round(e2 / S["kw_ac"]),
               "Kapazitätsfaktor AC": round(e2 / S["kw_ac"] / (8760 + 24 * (yr % 4 == 0)), 3)}
        if S["v1"] is not None:
            e1 = float(S["v1"].loc[str(yr)].sum()) * S["step_h"] / 1e3
            row.update({"v1 kWh/kWp": round(e1 / S["kwp"]), f"{S['name'][-2:]}/v1": round(e2 / e1, 4)})
        y.append(row)
    pk = df.power_park.max() / (S["kw_ac"] * 1e3)
    parts += [f"<h2>{n}. Ertrag</h2>", PR.table(pd.DataFrame(y)),
              f"<p class='note'>Höchste Stundenleistung {pk:.3f} · P_AC; Clipping-Stunden {100 * float((df.clipped > 0).mean()):.2f} % aller Stunden; "
              f"Schneeverlust {100 * float(park.snow_loss_share):.2f} % der Energie (gesamter Zeitraum).</p>",
              fig_monthly(S), f"<p class='note'>Leistung je Stunde und Tag ({yrs[-1] if yrs else ''}), normiert auf P_AC; Lücken im Winter = Schnee.</p>",
              fig_heatmap(S, yrs[-1] if yrs else df.index.year[0])]
    n += 1

    # example weeks
    summer = df.loc[(df.index.month.isin([5, 6, 7]))]
    e_wk = summer.power_park.resample("W-MON", label="left", closed="left").sum()
    ws = e_wk.idxmax()
    lost_wk = (df.power_park_free - df.power_park).resample("W-MON", label="left", closed="left").sum()
    sw = lost_wk.idxmax() if lost_wk.max() > 0 else df.loc[df.index.month == 1].power_park.resample("W-MON", label="left", closed="left").sum().idxmax()
    parts += [f"<h2>{n}. Beispielwochen</h2>", fig_week(S, ws, f"ertragsreichste Woche Mai–Juli ab {ws:%Y-%m-%d} (UTC)"),
              fig_week(S, sw, f"Woche mit dem größten Schneeverlust ab {sw:%Y-%m-%d} (UTC)" if lost_wk.max() > 0 else f"Januarwoche ab {sw:%Y-%m-%d} (kein Schnee)", snow=True)]
    n += 1

    # snow
    wt = winters(df, S["step_h"])
    parts += [f"<h2>{n}. Schnee-Modellierung</h2>",
              "<p>Release-Modell <b>depth_linear</b>: Bedeckung = clip(ERA5-Schneehöhe / 5 cm, 0, 1) je Stunde aus der nächsten ERA5-Zelle; "
              "Leistungsfaktor = 1 − pvlib dc_loss_nrel(Bedeckung, 100 Strings) ≈ 1 − Bedeckung; wirkt multiplikativ auf die AC-Leistung "
              "(power_park = power_park_free · snow_factor). Kein Abrutschen, kein Abtauen über die Modultemperatur: der Park bleibt bedeckt, "
              "solange ERA5 Schnee meldet. Die Variante depth_slide (Abrutschen nach Neigung und Temperatur) ist implementiert, aber nicht im Release (Entscheidung D7 offen).</p>",
              "<div style='display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start'>" + fig_snow_curve() + PR.table(wt) + "</div>",
              fig_snow_winters(S)]
    n += 1

    # drivers
    d = S["drv"]
    dy = []
    for yr in yrs:
        s = d.loc[str(yr)]
        dy.append({"Jahr": yr, "GHI CAMS roh (kWh/m²)": round(float(s.ghi_cams_raw.sum()) * S["step_h"] / 1e3), "GHI korrigiert": round(float(s.ghi.sum()) * S["step_h"] / 1e3),
                   "GHI ERA5": round(float(s.ghi_era5.sum()) * S["step_h"] / 1e3), "T mittel (°C)": round(float(s.temp_air.mean()), 1),
                   "Wind mittel (m/s)": round(float(s.wind_speed.mean()), 1), "Albedo Mittel / max": f"{s.albedo.mean():.2f} / {s.albedo.max():.2f}",
                   "Stunden mit Schnee > 0": int((s.snow_depth_cm > 0.05).sum())})
    parts += [f"<h2>{n}. Treiber</h2>", PR.table(pd.DataFrame(dy)),
              "<details><summary>Parameter des Manifests (JSON)</summary><pre style='white-space:pre-wrap;font-size:11px'>"
              + html.escape(json.dumps({k: man[k] for k in ("params", "config")}, indent=1, default=str)) + "</pre></details>"]
    return parts
