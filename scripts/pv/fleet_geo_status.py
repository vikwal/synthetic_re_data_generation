#!/usr/bin/env python
"""Fleet summary of the angle estimation of a release: one row per park with the chosen source, survey year, park
acceptance, main group and register comparison, the arrays the generator gets, and the LaTeX fragment of the paper
table (accepted measurements only). Reads the estimation store (parks_pv.geometry_store, geometry.method of the run
config) and the arrays of stage geometry (data/<release>/geometry_arrays.csv).

    synthre/bin/python scripts/pv/fleet_geo_status.py --config configs/runs/parks_pv_v4.yaml [--out ~/Work/pv_sim2real/results/geodata_release_<date>]
Outputs (no coordinates): fleet_geo_status_v2.csv, all_sources_v2.csv, pv_arrays_geodata_v2.csv, tab_geofleet_parks.tex."""
import argparse
import os
import sys

import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parks_pv import geometry_store as GSt, run_config, stages  # noqa: E402

STATE_EN = {"Bayern": "Bavaria", "Niedersachsen": "Lower Saxony", "Mecklenburg-Vorpommern": "Mecklenburg-W. P.", "Brandenburg": "Brandenburg",
            "Nordrhein-Westfalen": "North Rhine-Westphalia", "Hessen": "Hesse", "Sachsen": "Saxony", "Thüringen": "Thuringia",
            "Sachsen-Anhalt": "Saxony-Anhalt", "Baden-Württemberg": "Baden-Württemberg", "Rheinland-Pfalz": "Rhineland-Palatinate",
            "Saarland": "Saarland", "Schleswig-Holstein": "Schleswig-Holstein"}
OR_EN = {"Süd": "south", "Süd-West": "south-west", "Süd-Ost": "south-east", "Ost-West": "east-west", "Ost": "east", "West": "west", "Nord": "north"}
TILT_EN = {"< 20 Grad": "below 20", "20 - 40 Grad": "20 to 40", "40 - 60 Grad": "40 to 60"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=os.path.expanduser(f"~/Work/pv_sim2real/results/geodata_release_{pd.Timestamp.now():%Y-%m-%d}"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rc = run_config.load(a.config)
    run_config.apply(rc)
    sel = stages.load_selection(rc)
    src = GSt.all_sources(rc["geometry"]["method"], list(sel.park_id))
    if not len(src):
        src = pd.DataFrame(columns=["pv", "source", "status", "status_reason", "source_rank", "best"])
    arr = pd.read_csv(stages.geometry_path(), dtype={"park_id": str})
    rows, arr_rows = [], []
    for r in sel.itertuples():
        s = src[src.pv == r.park_id]
        best = s[s.best.astype(bool)] if "best" in s and len(s) else s.iloc[:0]
        pick = best.iloc[0] if len(best) else (s.sort_values("source_rank").iloc[0] if len(s) and s.source.notna().any() else None)
        a_ = arr[arr.park_id == r.park_id]
        geo_src = a_.geometry_source.iloc[0]
        arr_rows += a_.to_dict("records")
        d = dict(park_id=r.park_id, bundesland=r.bundesland, dc_kwp=r.dc_kwp, year=r.year, orientation_class=r.orientation_class, tilt_class=r.tilt_class,
                 cls=r.cls, n_sources=int(s.source.notna().sum()) if len(s) else 0, geometry_source=geo_src,
                 n_arrays=len(a_), arrays="; ".join(f"{x.tilt:.1f}/{x.azimuth:.0f}x{x.share:.2f}" for x in a_.itertuples()))
        if pick is not None:
            d.update(source=pick.source, status=pick.get("status"), status_reason=pick.get("status_reason"), footprint=pick.get("footprint"),
                     survey_year=pick.get("survey_year"), gate_after_build=pick.get("gate_after_build"), accept=pick.get("accept"),
                     accept_relaxed=pick.get("accept_relaxed"), n_windows=pick.get("n_windows"), n_tab_windows=pick.get("n_tab_windows"),
                     n_ok=pick.get("n_ok"), tab_share_accepted=pick.get("tab_share_accepted"), n_groups=pick.get("n_groups"),
                     tilt_measured=pick.get("tilt_measured"), tilt_p10=pick.get("g0_tilt_p10"), tilt_p90=pick.get("g0_tilt_p90"),
                     tilt_ends=pick.get("tilt_ends_med"), tilt_rows=pick.get("tilt_rows_med"), az_measured=pick.get("az_measured"),
                     az_source=pick.get("g0_az_source"), ew_measured=pick.get("ew_measured"), pitch=pick.get("g0_pitch"),
                     or_cls_measured=pick.get("or_cls_measured"), tilt_cls_measured=pick.get("tilt_cls_measured"), or_agree=pick.get("or_agree"),
                     tilt_agree=pick.get("tilt_agree"), tilt_agree_tol=pick.get("tilt_agree_tol"), warn_rejected=pick.get("warn_rejected"),
                     rej_tab_frac=pick.get("rej_tab_frac"))
        else:
            d.update(source=None, status="no_fetcher" if not len(s) else "failed", status_reason=s.status_reason.iloc[0] if len(s) else "")
        rows.append(d)
    st = pd.DataFrame(rows)
    st.to_csv(os.path.join(a.out, "fleet_geo_status_v2.csv"), index=False)
    src.drop(columns=[c for c in src.columns if c in ("survey_dates",)]).to_csv(os.path.join(a.out, "all_sources_v2.csv"), index=False)
    pd.DataFrame(arr_rows).to_csv(os.path.join(a.out, "pv_arrays_geodata_v2.csv"), index=False)
    # LaTeX fragment: accepted measurements, by state and year
    acc = st[st.accept.fillna(False).astype(bool)].sort_values(["bundesland", "year"])
    lines = [r"\begin{tabular}{@{}rlllllllr@{}}", r"\toprule",
             r" & state & built & survey & source & register orientation & register tilt & measured azimuth & measured tilt \\", r"\midrule"]
    for k, x in enumerate(acc.itertuples(), 1):
        az = "east-west" if bool(x.ew_measured) else f"{x.az_measured:.0f}"
        sy = "" if pd.isna(x.survey_year) else f"{int(x.survey_year)}"
        srcn = {"laz": "point cloud", "dom1_laser": "DOM1", "bdom_image": "image DSM", "dom20_image": "image DSM"}.get(x.source, x.source)
        lines.append(f"{k} & {STATE_EN.get(x.bundesland, x.bundesland)} & {x.year} & {sy} & {srcn} & {OR_EN.get(x.orientation_class, x.orientation_class)} & "
                     f"{TILT_EN.get(x.tilt_class, x.tilt_class)} & {az} & {x.tilt_measured:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(a.out, "tab_geofleet_parks.tex"), "w").write("\n".join(lines) + "\n")
    # summary for the README
    n = len(st)
    run = st[st.status != "no_fetcher"]
    print(f"{n} parks; with a fetcher {len(run)}; status {run.status.value_counts().to_dict()}; accepted {int(acc.shape[0])} "
          f"(by source {acc.source.value_counts().to_dict()}); relaxed would accept {int(run.accept_relaxed.fillna(False).astype(bool).sum())}")
    if len(acc):
        om = acc[acc.az_source.isin(["measured", "axis"])]
        print(f"orientation class agrees {int(om.or_agree.astype(bool).sum())}/{len(om)}; tilt class {int(acc.tilt_agree.astype(bool).sum())}/{len(acc)} "
              f"(±2°: {int(acc.tilt_agree_tol.astype(bool).sum())}/{len(acc)})")
    print("geometry source:", st.geometry_source.str.split(":").str[0].value_counts().to_dict())
    print("written", a.out)


if __name__ == "__main__":
    main()
