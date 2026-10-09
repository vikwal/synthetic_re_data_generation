"""Validation of a curtailed release (stage curt_validate): metrics JSON/CSV and figures
for FL_Contribution/reports/curtailment_synthesis_v1.md.

Sources of the reference values: curtailment/validate.py (rows of calibration_numbers.md).
Output: FL_Contribution/reports/figs_curtailment_synthesis_v1/{metrics_<dataset>.json, *.png, *.csv}
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from curtailment import calibrate, drivers, grid, timegrid
from curtailment import validate as V
from curtailment.config import resolve_path
from parks import curtail, paths

FIG_DIR = os.path.join(paths.FL_DIR, "reports", "figs_curtailment_synthesis_v1")
FL_CURT = os.path.join(paths.FL_DIR, "pipeline", "curtailment")
NETZAMPEL = os.path.join(paths.FL_DIR, "data", "curtailment", "raw", "netzampel_eon")
# palette (dataviz reference, light mode): categorical slots 1-3, neutral reference gray, ink
BLUE, ORANGE, AQUA, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#9a9893", "#0b0b0b", "#52514e"
AREA_LABEL = {"A1_SH": "A1 SH", "A2_NI_NW": "A2 NI-NW", "A3_NI_O_ST": "A3 NI-O/ST", "A4_NO": "A4 NO",
              "A5_MITTE_W": "A5 Mitte/W", "A6_SUED": "A6 Süd"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRAY, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "figure.dpi": 150, "savefig.bbox": "tight", "axes.axisbelow": True})
    return plt


# ---------------------------------------------------------------- data

def _load_release(cfg) -> dict:
    out = curtail.release_dir(cfg)
    parks = pd.read_csv(os.path.join(out, "parks.csv"))
    frames = {p: pd.read_parquet(os.path.join(out, f"synth_{p}.parquet"),
                                 columns=["power_park", "power_park_avail", "grid_setpoint", "curt_flag",
                                          "loss_env", "loss_mkt", "loss_grid", "neg_block_len_h", "grid_event_id"])
              for p in parks["park_id"]}
    return {"parks": parks, "frames": frames, "summary": pd.read_csv(os.path.join(out, "curtailment_summary.csv")),
            "events": pd.read_csv(os.path.join(out, "grid_events.csv")) if os.path.exists(
                os.path.join(out, "grid_events.csv")) else pd.DataFrame(),
            "clients": pd.read_csv(os.path.join(out, "clients.csv"))}


def _fleet_job(args):
    cfg, area, qidx, cf, u, prof, c_by_year = args
    prob = calibrate.AreaGridProblem(cfg, area, qidx, cf, u, prof)
    d = calibrate.fleet_detail(prob, c_by_year, cf, V.COUPLING_BINS, V.START_BINS)
    return area, d


def fleet_stats(chain, cfg, calib, workers) -> dict:
    idx = curtail.hourly_index(chain)
    qidx = timegrid.quarter_index(idx)
    drv, _ = drivers.load(curtail.driver_cache(cfg), qidx)
    parks, _ = curtail.source_tables()
    nodes = curtail.node_table(cfg, parks)
    prof = curtail.area_profiles(parks, nodes, len(idx))
    areas = [a for a, v in cfg["grid"]["areas"].items() if v["p0"] < 1]
    u = grid.u_on_slots(qidx, areas, cfg["grid"]["disturbance"], cfg["seed"])
    jobs = [(cfg, a, qidx, drv["cf_da"].to_numpy(), u[a], prof[a][1],
             {int(y): v["c"] for y, v in calib["grid"][a].items()}) for a in areas]
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        return dict(ex.map(_fleet_job, jobs)), drv


# ---------------------------------------------------------------- metrics

def energy_metrics(rel, calib, cfg) -> dict:
    summ = rel["summary"].merge(rel["parks"][["park_id", "curt_area", "client_id"]], on="park_id")
    w = summ.assign(e_grid=summ["share_grid"] * summ["energy_avail_mwh"])
    by_ay = w.groupby(["curt_area", "year"])[["e_grid", "energy_avail_mwh"]].sum()
    parks_share = (100 * by_ay["e_grid"] / by_ay["energy_avail_mwh"]).unstack()
    tgt = {a: {y: cfg["grid"]["target_scale"] * v for y, v in cfg["grid"]["areas"][a].get(
        "target_pct", {}).items()} for a in cfg["grid"]["areas"]}
    virt = {a: {int(y): 100 * r.get("achieved", 0.0) for y, r in calib["grid"][a].items()} for a in calib["grid"]}
    gas = json.load(open(os.path.join(FL_CURT, "grid_area_summary.json")))
    fe = {curtail_area(k): v for k, v in gas["fleet_energy_gwh_assumed"].items()}
    fleet_sum = {}
    for y in (2023, 2024, 2025, 2026):
        num = sum(fe[a] * virt.get(a, {}).get(y, 0.0) for a in fe)
        fleet_sum[y] = num / sum(fe.values())
    exp = gas["client_expected_share_pct_mw_weighted"]
    wc = w.groupby(["client_id", "year"])[["e_grid", "energy_avail_mwh"]].sum()
    client_share = (100 * wc["e_grid"] / wc["energy_avail_mwh"]).unstack()
    return {"parks_area_year_pct": parks_share.round(3).to_dict(orient="index"), "target_pct": tgt,
            "virtual_area_year_pct": virt, "virtual_fleet_weighted_pct": fleet_sum,
            "client_year_pct": client_share.round(3).to_dict(orient="index"),
            "client_expected_pct": {y: {k: v * cfg["grid"]["target_scale"] for k, v in d.items()}
                                    for y, d in exp.items()}}


def curtail_area(name: str) -> str:
    from curtailment.config import area_key
    return area_key(name)


def hours_metrics(rel, fleet) -> dict:
    out = {}
    for a, d in fleet.items():
        h = d["hours"]
        full = h[h["year"].isin([2024, 2025])]
        aff = full[full["hours"] > 0]
        out[a] = {"virtual_affected": V.quantiles(aff["hours"]), "virtual_share_affected": float(
            len(aff) / len(full)) if len(full) else np.nan, "reference": V.REF_HOURS[a]}
    summ = rel["summary"].merge(rel["parks"][["park_id", "curt_area"]], on="park_id")
    s = summ[summ["year"].isin([2024, 2025]) & (summ["hours_grid_setpoint"] > 0)]
    for a, g in s.groupby("curt_area"):
        out.setdefault(a, {})["parks_affected"] = V.quantiles(g["hours_grid_setpoint"])
        out[a]["parks_n_park_years"] = int(len(g))
    return out


def sh_metrics(rel, cfg) -> dict:
    parks = rel["parks"]
    sh = parks[parks["curt_area"] == "A1_SH"]
    flags = pd.DataFrame({p: (rel["frames"][p]["grid_setpoint"] < 1).astype(int) for p in sh["park_id"]})
    run_l, gap_l = [], []
    for p in flags:
        r, g = V.runs(flags[p].to_numpy())
        run_l.append(r)
        gap_l.append(g)
    run_l, gap_l = np.concatenate(run_l), np.concatenate(gap_l)
    lat, lon = sh.set_index("park_id")["latitude"], sh.set_index("park_id")["longitude"]
    phi = V.phi_by_distance(flags, lat, lon)
    phi_bin = phi.groupby("bin", observed=False)["phi"].agg(["mean", "count"]).reset_index() if len(phi) else None
    # per park vs Netzampel commune flags (upper bound: commune flag = any unit, all technologies)
    pin = pd.read_csv(os.path.join(NETZAMPEL, "parks_in_netzampel.csv"), dtype={"ags": str})
    pin = pin[pin["op"] == "shn"].rename(columns={pin.columns[0]: "park_id"})
    hf = pd.read_csv(os.path.join(NETZAMPEL, "hourly_flags_2023_2025.csv.gz"), index_col=0)
    hf.index = pd.to_datetime(hf.index, utc=True)
    fix = _fix_commune(pin, parks)
    rows = []
    for r in pin.itertuples(index=False):
        ags = fix.get(r.park_id, r.ags)
        f = flags[r.park_id] if r.park_id in flags else None
        if f is None:
            continue
        common = f.index.intersection(hf.index)
        commune = hf.loc[common, ags].mean() if ags in hf.columns else np.nan
        rows.append({"park_id": r.park_id, "commune": r.commune if ags == r.ags else f"{ags} (corrected)",
                     "ags": ags, "park_flag_pct": 100 * f.loc[common].mean(),
                     "commune_flag_pct": 100 * commune, "node": parks.set_index("park_id").loc[r.park_id, "curt_node"],
                     "node_free_p0": bool(parks.set_index("park_id").loc[r.park_id, "curt_node_free_p0"])})
    tab = pd.DataFrame(rows)
    mon = flags.groupby(flags.index.month).mean().mean(axis=1) * 100
    return {"runs": {**V.quantiles(run_l, (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)), "mean": float(run_l.mean()) if len(
        run_l) else np.nan, "n": int(len(run_l))}, "gaps": {**V.quantiles(gap_l, (0.25, 0.5, 0.75, 0.9)),
                                                           "n": int(len(gap_l))},
            "phi_by_distance": phi_bin.to_dict(orient="records") if phi_bin is not None else [],
            "flag_share_pct": float(100 * flags.values.mean()), "monthly_flag_pct": mon.round(2).to_dict(),
            "parks_vs_netzampel": tab.round(2).to_dict(orient="records"), "_phi": phi, "_runs": run_l}


def _fix_commune(pin: pd.DataFrame, parks: pd.DataFrame) -> dict:
    """Parks whose release coordinate (parks.csv, corrected) is > 1 km from the one used for
    parks_in_netzampel.csv (MaStR placeholder coordinates) get the nearest SH-Netz commune
    centroid instead (approximation; reported)."""
    from curtailment.areas import haversine_km
    cen = {k: v for k, v in json.load(open(os.path.join(NETZAMPEL, "centroids_all.json"))).items() if v[3] == "shn"}
    pos = parks.set_index("park_id")[["latitude", "longitude"]]
    out = {}
    for r in pin.itertuples(index=False):
        if r.park_id not in pos.index:
            continue
        lat, lon = pos.loc[r.park_id]
        if haversine_km(lat, lon, r.lat, r.lon) > 1.0:
            out[r.park_id] = min(cen, key=lambda k: haversine_km(lat, lon, cen[k][1], cen[k][2]))
    return out


def market_metrics(rel, calib, drv, cfg) -> dict:
    ach = calib["market"]["achieved"]
    regime = pd.read_csv(os.path.join(FL_CURT, "park_grid_areas.csv")).set_index("lokation")["regime_2025"]
    rows = []
    for p, f in rel["frames"].items():
        neg = f["neg_block_len_h"] > 0
        av = f.loc[neg, "power_park_avail"].sum()
        rows.append({"park_id": p, "regime_2025": regime.get(p, ""), "avail_neg_mwh": av / 1e6,
                     "loss_mkt_neg_mwh": f.loc[neg, "loss_mkt"].sum() / 1e6,
                     "share_mkt_total": f["loss_mkt"].sum() / f["power_park_avail"].sum()})
    t = pd.DataFrame(rows)
    by_reg = t.groupby("regime_2025").agg(n=("park_id", "size"), avail=("avail_neg_mwh", "sum"),
                                         loss=("loss_mkt_neg_mwh", "sum"))
    by_reg["rate_neg_pct"] = 100 * by_reg["loss"] / by_reg["avail"]
    rm = drivers.monthly_market_value(curtail.driver_cache(cfg))
    rm_cmp = []
    for per, v in rm.items():
        ref = V.REF_RM.get(per.year)
        rm_cmp.append({"month": str(per), "r_m": float(v), "ref_D7b": ref[per.month - 1] if ref else np.nan})
    rmc = pd.DataFrame(rm_cmp)
    both = rmc.dropna()
    return {"virtual": ach, "theta": {k: calib["market"][k] for k in ("theta0", "theta1", "tau")},
            "fit_rmse": calib["market"].get("rmse"), "parks_by_regime": by_reg.round(3).reset_index().to_dict(
                orient="records"), "parks_rate_neg_pct_all": float(100 * t["loss_mkt_neg_mwh"].sum()
                                                                    / t["avail_neg_mwh"].sum()),
            "r_m": rmc.round(2).to_dict(orient="records"),
            "r_m_vs_D7b": {"n": len(both), "mean_abs_diff": float((both["r_m"] - both["ref_D7b"]).abs().mean()),
                           "mean_diff": float((both["r_m"] - both["ref_D7b"]).mean()),
                           "max_abs_diff": float((both["r_m"] - both["ref_D7b"]).abs().max())}}


def bat_metrics(rel) -> dict:
    s = rel["summary"].merge(rel["parks"][["park_id", "curt_area", "curt_bat_level"]], on="park_id")
    s = s[s["year"].isin([2024, 2025]) & s["curt_bat_level"].notna() & (s["curt_bat_level"] != "")]
    x = 100 * s["share_env"]
    out = {"n_park_years": int(len(s)), "n_parks": int(s["park_id"].nunique()), "median_pct": float(x.median()),
           "share_le_2pct": float((x <= 2).mean()), "share_lt_5pct": float((x < 5).mean()),
           "p90_pct": float(x.quantile(0.9)), "max_pct": float(x.max()),
           "by_area_median_pct": s.assign(x=x).groupby("curt_area")["x"].median().round(2).to_dict(),
           "by_level_median_pct": s.assign(x=x).groupby("curt_bat_level")["x"].median().round(2).to_dict(),
           "parks_with_permit": int((rel["parks"]["curt_bat_level"].fillna("") != "").sum())}
    out["_x"] = x.to_numpy()
    return out


def detection_metrics(rel) -> dict:
    hub = {}
    rows = []
    for p, f in rel["frames"].items():
        src = curtail.read_source(p)
        gids = [c[len("power_"):] for c in src.columns if c.startswith("power_t")]
        w = sum(src[f"power_{g}"].max() for g in gids)
        wind = sum(src[f"wind_speed_hub_{g}"] * src[f"power_{g}"].max() for g in gids) / w
        cap = rel["parks"].set_index("park_id").loc[p, "capacity_kw"] * 1000
        flag = V.detector(f["power_park"], wind, cap)
        loss = f["power_park_avail"] - f["power_park"]
        sc = V.detection_scores(flag, f["curt_flag"] == 1, loss)
        sc_grid = V.detection_scores(flag, f["loss_grid"] > 0, f["loss_grid"])
        rows.append({"park_id": p, **sc, "recall_energy_grid": sc_grid["recall_energy"],
                     "loss_mwh": loss.sum() / 1e6, "loss_grid_mwh": f["loss_grid"].sum() / 1e6,
                     "detected_loss_mwh": loss[flag].sum() / 1e6, "detected_grid_mwh": f["loss_grid"][flag].sum() / 1e6,
                     "tp": int((flag & (f["curt_flag"] == 1)).sum())})
    t = pd.DataFrame(rows)
    return {"pooled_recall_energy": float(t["detected_loss_mwh"].sum() / t["loss_mwh"].sum()),
            "pooled_recall_energy_grid": float(t["detected_grid_mwh"].sum() / t["loss_grid_mwh"].sum())
            if t["loss_grid_mwh"].sum() > 0 else np.nan,
            "pooled_precision": float(t["tp"].sum() / t["hours_flagged"].sum()),
            "pooled_recall_hours": float(t["tp"].sum() / t["hours_true"].sum()),
            "hours_flagged": int(t["hours_flagged"].sum()), "hours_true": int(t["hours_true"].sum())}


def consistency(rel, cfg) -> dict:
    worst_excess, worst_sum = 0.0, 0.0
    for f in rel["frames"].values():
        worst_excess = max(worst_excess, float((f["power_park"] - f["power_park_avail"]).max()))
        d = (f["loss_env"] + f["loss_mkt"] + f["loss_grid"]) - (f["power_park_avail"] - f["power_park"])
        worst_sum = max(worst_sum, float(d.abs().max()))
    out = {"max_power_minus_avail_W": worst_excess, "max_abs_loss_sum_error_W": worst_sum}
    other = "parks_v1_curt" if cfg["dataset"].endswith("_x4") else None
    if other:
        p = os.path.join(paths.DATA_ROOT, "synthetic", "wind", other, "grid_events.csv")
        if os.path.exists(p) and len(rel["events"]):
            e1 = set(pd.read_csv(p)["event_id"])
            e4 = set(rel["events"]["event_id"])
            out["x4_superset_of_x1"] = {"n_x1": len(e1), "n_x4": len(e4), "missing_in_x4": len(e1 - e4)}
    return out


def client_table(rel) -> pd.DataFrame:
    c = rel["clients"]
    cols = ["client_id", "n_parks", "capacity_kw", "curt_share_env", "curt_share_mkt", "curt_share_grid",
            "curt_share_total", "curt_flag_hours_share_mean", "curt_parks_with_grid_events"]
    return c[cols]


# ---------------------------------------------------------------- figures

def fig_area_shares(m, ds, plt):
    yrs = [2024, 2025]
    areas = [a for a in AREA_LABEL if a != "A6_SUED"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    x = np.arange(len(areas))
    for ax, y in zip(axs, yrs):
        t = [m["target_pct"][a].get(y, np.nan) for a in areas]
        v = [m["virtual_area_year_pct"][a].get(y, np.nan) for a in areas]
        p = [m["parks_area_year_pct"].get(a, {}).get(y, np.nan) for a in areas]
        ax.bar(x - 0.27, t, 0.25, color=GRAY, label="Ziel [C53]")
        ax.bar(x, v, 0.25, color=BLUE, label="virtuelle Flotte")
        ax.bar(x + 0.27, p, 0.25, color=ORANGE, label="unsere Parks")
        ax.set_xticks(x, [AREA_LABEL[a] for a in areas], rotation=30, ha="right")
        ax.set_title(str(y), color=INK)
    axs[0].set_ylabel("netzbedingt abgeregelt [% der Energie]")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07))
    fig.savefig(os.path.join(FIG_DIR, f"area_shares_{ds}.png"))
    plt.close(fig)


def fig_coupling(fleet, ds, plt):
    areas = list(fleet)
    fig, axs = plt.subplots(1, len(areas), figsize=(2.2 * len(areas), 2.6), sharey=False)
    mids = [np.mean(b) for b in V.COUPLING_BINS]
    for ax, a in zip(np.atleast_1d(axs), areas):
        ax.plot(mids, V.REF_COUPLING[a], "o--", color=GRAY, lw=1.5, ms=5, label="Referenz [C19]")
        ax.plot(mids, fleet[a]["coupling_pct"], "o-", color=BLUE, lw=2, ms=5, label="virtuelle Flotte")
        ax.set_title(AREA_LABEL[a], color=INK)
        ax.set_xlabel("CF_DA")
    np.atleast_1d(axs)[0].set_ylabel("tiefengew. Anteil [%]")
    np.atleast_1d(axs)[0].legend(loc="upper left", fontsize=7)
    fig.savefig(os.path.join(FIG_DIR, f"coupling_{ds}.png"))
    plt.close(fig)


def fig_hours(hm, ds, plt):
    areas = [a for a in AREA_LABEL if a in hm and "virtual_affected" in hm[a]]
    fig, ax = plt.subplots(figsize=(6.5, 3))
    for i, a in enumerate(areas):
        r = hm[a]["reference"]
        v = hm[a]["virtual_affected"]
        ax.plot([i - 0.15] * 2, [r[1], r[3]], color=GRAY, lw=2)
        ax.plot(i - 0.15, r[2], "o", color=GRAY, ms=7, label="Referenz p10–p90, p50" if i == 0 else None)
        ax.plot([i + 0.15] * 2, [v["p10"], v["p90"]], color=BLUE, lw=2)
        ax.plot(i + 0.15, v["p50"], "o", color=BLUE, ms=7, label="virtuelle Flotte" if i == 0 else None)
        if "parks_affected" in hm[a]:
            ax.plot(i + 0.35, hm[a]["parks_affected"]["p50"], "D", color=ORANGE, ms=6,
                    label="unsere Parks p50" if i == 0 else None)
    ax.set_yscale("log")
    ax.set_xticks(range(len(areas)), [f"{AREA_LABEL[a]}\n{hm[a]['reference'][0]}" for a in areas], fontsize=7)
    ax.set_ylabel("abgeregelte h je betroff. Einheit und Jahr")
    ax.legend(fontsize=7, loc="lower left")
    fig.savefig(os.path.join(FIG_DIR, f"hours_{ds}.png"))
    plt.close(fig)


def fig_sh(sh, ds, plt):
    fig, axs = plt.subplots(1, 2, figsize=(8, 2.8))
    ks = ["p10", "p25", "p50", "p75", "p90", "p99"]
    axs[0].plot(range(len(ks)), [V.REF_RUNS[k] for k in ks], "o--", color=GRAY, lw=1.5, label="Netzampel [C40]")
    axs[0].plot(range(len(ks)), [sh["runs"][k] for k in ks], "o-", color=BLUE, lw=2, label="SH-Parks")
    axs[0].set_xticks(range(len(ks)), ks)
    axs[0].set_yscale("log")
    axs[0].set_ylabel("Lauflänge [h]")
    axs[0].legend(fontsize=7)
    pb = pd.DataFrame(sh["phi_by_distance"])
    lab = list(V.REF_PHI)
    axs[1].plot(range(len(lab)), [V.REF_PHI[k] for k in lab], "o--", color=GRAY, lw=1.5, label="Netzampel [C43]")
    if len(pb):
        axs[1].plot(range(len(lab)), pb.set_index("bin").reindex(lab)["mean"], "o-", color=BLUE, lw=2,
                    label="SH-Parks")
    axs[1].set_xticks(range(len(lab)), lab)
    axs[1].set_xlabel("Abstand [km]")
    axs[1].set_ylabel("φ")
    axs[1].legend(fontsize=7)
    fig.savefig(os.path.join(FIG_DIR, f"sh_runs_phi_{ds}.png"))
    plt.close(fig)


def fig_market(mm, cfg, ds, plt):
    bands = list(cfg["market"]["response"]["target_band_rate"])
    fig, axs = plt.subplots(1, 2, figsize=(8, 2.8))
    x = np.arange(len(bands))
    axs[0].plot(x, [100 * cfg["market"]["response"]["target_band_rate"][b] for b in bands], "o--", color=GRAY,
                lw=1.5, label="Hirth [D3]")
    axs[0].plot(x, [100 * mm["virtual"]["band_rate"][b] for b in bands], "o-", color=BLUE, lw=2,
                label="virtuelle Flotte")
    axs[0].set_xticks(x, bands, fontsize=7)
    axs[0].set_xlabel("DA-Preis [€/MWh]")
    axs[0].set_ylabel("abgeregelt [% der pot.]")
    axs[0].legend(fontsize=7)
    yrs = sorted(int(y) for y in mm["virtual"]["fleet_rate_neg"])
    axs[1].plot(yrs, [V.REF_D4[y][2] for y in yrs], "o--", color=GRAY, lw=1.5, label="Hirth [D4]")
    axs[1].plot(yrs, [100 * mm["virtual"]["fleet_rate_neg"][str(y) if str(y) in mm["virtual"]["fleet_rate_neg"]
                                                            else y] for y in yrs], "o-", color=BLUE, lw=2,
                label="virtuelle Flotte")
    axs[1].axhline(8.7, color=ORANGE, lw=1, ls=":", label="Untergrenze 8,7 % [D9]")
    axs[1].set_xticks(yrs, [str(y) for y in yrs])
    axs[1].set_ylabel("Rate in neg. Stunden [%]")
    axs[1].legend(fontsize=7)
    fig.savefig(os.path.join(FIG_DIR, f"market_{ds}.png"))
    plt.close(fig)


def fig_bat(bm, ds, plt):
    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    ax.hist(bm["_x"], bins=np.arange(0, max(8, bm["_x"].max() + 0.5), 0.25), color=BLUE, edgecolor="white", lw=0.5)
    ax.axvline(2, color=GRAY, ls="--", lw=1)
    ax.axvline(5, color=GRAY, ls=":", lw=1)
    ax.set_xlabel("Jahresverlust Fledermaus [% der Energie]")
    ax.set_ylabel("Park-Jahre")
    fig.savefig(os.path.join(FIG_DIR, f"bat_{ds}.png"))
    plt.close(fig)


def fig_clients(ct, ds, plt):
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    x = np.arange(len(ct))
    bottom = np.zeros(len(ct))
    for col, color, lab in (("curt_share_grid", BLUE, "Netz"), ("curt_share_mkt", ORANGE, "Markt"),
                            ("curt_share_env", AQUA, "Umwelt")):
        v = 100 * ct[col].to_numpy()
        ax.bar(x, v, 0.6, bottom=bottom, color=color, label=lab, edgecolor="white", lw=1)
        bottom += v
    ax.set_xticks(x, ct["client_id"])
    ax.set_ylabel("Energieverlust [%]")
    ax.legend(ncol=3, fontsize=7, loc="upper right")
    fig.savefig(os.path.join(FIG_DIR, f"clients_{ds}.png"))
    plt.close(fig)


def fig_example(rel, ds, plt, park: str = None):
    """One week of a park (default: largest grid loss share) with its largest weekly grid loss."""
    if park is None:
        park = max(rel["frames"], key=lambda p: rel["frames"][p]["loss_grid"].sum()
                   / max(rel["frames"][p]["power_park_avail"].sum(), 1.0))
    f = rel["frames"][park]
    cap = rel["parks"].set_index("park_id").loc[park, "capacity_kw"] * 1000
    wk = f["loss_grid"].resample("7D").sum().idxmax()
    w = f.loc[wk:wk + pd.Timedelta(days=7)]
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.plot(w.index, w["power_park_avail"] / cap, color=GRAY, lw=1.5, label="verfügbar (power_park_avail)")
    ax.plot(w.index, w["power_park"] / cap, color=BLUE, lw=2, label="tatsächlich (power_park)")
    ax.step(w.index, w["grid_setpoint"], where="post", color=ORANGE, lw=1, ls="--", label="Netz-Sollwert (Stundenmittel)")
    ax.set_ylabel("Leistung / P_inst")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    fig.autofmt_xdate()
    fig.savefig(os.path.join(FIG_DIR, f"example_{ds}.png"))
    plt.close(fig)


# ---------------------------------------------------------------- run

def _clean(d):
    if isinstance(d, dict):
        return {str(k): _clean(v) for k, v in d.items() if not str(k).startswith("_")}
    if isinstance(d, (list, tuple)):
        return [_clean(v) for v in d]
    if isinstance(d, (np.floating, np.integer)):
        return d.item()
    if isinstance(d, float) and np.isnan(d):
        return None
    return d


def run(chain, cfg, workers: int = 8) -> dict:
    os.makedirs(FIG_DIR, exist_ok=True)
    ds = cfg["dataset"]
    calib = json.load(open(curtail.calibration_path(cfg)))
    rel = _load_release(cfg)
    fleet, drv = fleet_stats(chain, cfg, calib, workers)
    m = {"dataset": ds, "target_scale": cfg["grid"]["target_scale"]}
    m["energy"] = energy_metrics(rel, calib, cfg)
    m["hours"] = hours_metrics(rel, fleet)
    m["fleet"] = {a: {k: v for k, v in d.items() if k != "hours"} for a, d in fleet.items()}
    m["coupling_reference"] = V.REF_COUPLING
    m["start_rate_reference"] = V.REF_START
    m["sh"] = sh_metrics(rel, cfg)
    m["market"] = market_metrics(rel, calib, drv, cfg)
    m["bat"] = bat_metrics(rel)
    m["detection"] = detection_metrics(rel)
    m["consistency"] = consistency(rel, cfg)
    ct = client_table(rel)
    m["clients"] = ct.round(5).to_dict(orient="records")
    ct.to_csv(os.path.join(FIG_DIR, f"clients_{ds}.csv"), index=False)
    plt = _plt()
    fig_area_shares(m["energy"], ds, plt)
    fig_coupling(fleet, ds, plt)
    fig_hours(m["hours"], ds, plt)
    fig_sh(m["sh"], ds, plt)
    fig_market(m["market"], cfg, ds, plt)
    fig_bat(m["bat"], ds, plt)
    fig_clients(ct, ds, plt)
    fig_example(rel, ds, plt)
    out = _clean(m)
    with open(os.path.join(FIG_DIR, f"metrics_{ds}.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps({k: out[k] for k in ("consistency", "detection")}, indent=1))
    print(ct.round(4).to_string(index=False))
    return out
