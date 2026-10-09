"""Driver series from the energy-charts API (Fraunhofer ISE), fetched once and
cached as Parquet; the only module of the package with I/O side effects.

Series (Germany, UTC):
  forecast_mw   day-ahead forecast wind onshore, 15 min   /public_power_forecast
  actual_mw     actual wind onshore, 15 min               /public_power
  price         day-ahead price DE-LU [EUR/MWh]; hourly until 2025-09-30,
                15 min from 2025-10-01 (local)            /price
  installed_gw  installed onshore capacity, yearly        /installed_power
Derived on the 15-min grid: cf_da = forecast / installed (installed linear
between year-end values), price piecewise constant within the hour, and the
monthly onshore market value r_m = sum(price * actual) / sum(actual)
(calendar months Europe/Berlin).

Licence: prices CC BY 4.0, Bundesnetzagentur | SMARD.de (per API response);
generation data energy-charts.info (Fraunhofer ISE), public data of the TSOs.
"""

import hashlib
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

from curtailment import timegrid

API = "https://api.energy-charts.info"
SERIES = ("forecast_mw", "actual_mw", "price")
LICENCE = {
    "price": "CC BY 4.0, Bundesnetzagentur | SMARD.de (energy-charts.info /price)",
    "forecast_mw": "energy-charts.info (Fraunhofer ISE), TSO day-ahead forecast wind onshore",
    "actual_mw": "energy-charts.info (Fraunhofer ISE), public net generation wind onshore",
    "installed_gw": "energy-charts.info (Fraunhofer ISE), installed power wind onshore (yearly)",
}
MAX_INTERP_H = 4.0   # gaps up to this length are interpolated (forecast/actual) or held (price)


def _get(url: str, retries: int = 4) -> dict:
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.loads(r.read().decode())
        except Exception:  # noqa: BLE001 - network errors: retry with backoff
            if k == retries - 1:
                raise
            time.sleep(5 * (k + 1))


def _url(series: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
    s, e = start.strftime("%Y-%m-%dT%H:%MZ"), end.strftime("%Y-%m-%dT%H:%MZ")
    if series == "forecast_mw":
        return (f"{API}/public_power_forecast?country=de&production_type=wind_onshore"
                f"&forecast_type=day-ahead&start={s}&end={e}")
    if series == "actual_mw":
        return f"{API}/public_power?country=de&start={s}&end={e}"
    if series == "price":
        return f"{API}/price?bzn=DE-LU&start={s}&end={e}"
    raise KeyError(series)


def _parse(series: str, js: dict) -> pd.Series:
    t = pd.to_datetime(np.asarray(js["unix_seconds"], dtype=np.int64), unit="s", utc=True)
    if series == "forecast_mw":
        v = js["forecast_values"]
    elif series == "price":
        v = js["price"]
    else:
        pt = {p["name"]: p["data"] for p in js["production_types"]}
        v = pt["Wind onshore"]
    return pd.Series(np.asarray([np.nan if x is None else x for x in v], dtype=float), index=t, name=series)


def month_windows(start, end) -> list:
    """Monthly UTC windows [m, next m - 15 min] covering [start, end]."""
    first = pd.Timestamp(start).tz_localize(None).to_period("M").to_timestamp().tz_localize("UTC")
    stop = pd.Timestamp(end, tz="UTC")
    out = []
    m = first
    while m <= stop:
        nxt = m + pd.offsets.MonthBegin(1)
        out.append((m, min(nxt - timegrid.STEP, stop)))
        m = nxt
    return out


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(cache: str, start, end, log=print) -> dict:
    """Fetch all series month by month into cache/raw, combine into
    cache/<series>.parquet and cache/installed_gw.json; write cache/manifest.json."""
    raw = os.path.join(cache, "raw")
    os.makedirs(raw, exist_ok=True)
    man = {"api": API, "window": [str(start), str(end)], "fetched": pd.Timestamp.now(tz="UTC").isoformat(
        timespec="seconds"), "licence": LICENCE, "files": {}, "gaps": {}}
    for series in SERIES:
        parts = []
        for a, b in month_windows(start, end):
            path = os.path.join(raw, f"{series}_{a:%Y-%m}.parquet")
            complete_month = b == a + pd.offsets.MonthBegin(1) - timegrid.STEP
            if os.path.exists(path) and complete_month:
                parts.append(pd.read_parquet(path)[series])
                continue
            s = _parse(series, _get(_url(series, a, b)))
            s.to_frame().to_parquet(path)
            parts.append(s)
            log(f"{series} {a:%Y-%m}: {len(s)} values, {int(s.isna().sum())} NaN")
        s = pd.concat(parts)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        path = os.path.join(cache, f"{series}.parquet")
        s.to_frame().to_parquet(path)
        man["files"][series] = {"path": os.path.basename(path), "sha256": sha256_file(path),
                                "n": len(s), "first": str(s.index[0]), "last": str(s.index[-1])}
    inst = _get(f"{API}/installed_power?country=de&time_step=yearly")
    pt = {p["name"]: p["data"] for p in inst["production_types"]}
    ip = {"time": inst["time"], "wind_onshore_gw": pt["Wind onshore"], "last_update": inst.get("last_update")}
    path = os.path.join(cache, "installed_gw.json")
    with open(path, "w") as f:
        json.dump(ip, f, indent=1)
    man["files"]["installed_gw"] = {"path": "installed_gw.json", "sha256": sha256_file(path)}
    with open(os.path.join(cache, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    return man


def installed_mw(cache: str, index: pd.DatetimeIndex) -> pd.Series:
    """Installed onshore capacity [MW], linear between year-end values; the value
    of the year of last_update is placed at last_update (the API reports the
    current state there, not a year-end value)."""
    with open(os.path.join(cache, "installed_gw.json")) as f:
        ip = json.load(f)
    upd = pd.Timestamp(int(ip["last_update"]), unit="s", tz="UTC") if ip.get("last_update") else None
    pts = {}
    for y, v in zip(ip["time"], ip["wind_onshore_gw"]):
        if v is None:
            continue
        t = pd.Timestamp(f"{int(y)}-12-31 23:00", tz="UTC")
        if upd is not None and int(y) == upd.year:
            t = upd
        pts[t] = float(v) * 1000.0
    cap = pd.Series(pts).sort_index()
    x = cap.index.asi8.astype(float)
    return pd.Series(np.interp(index.asi8.astype(float), x, cap.values), index=index, name="installed_mw")


def _on_grid(s: pd.Series, qidx: pd.DatetimeIndex, hold: bool) -> tuple:
    """Series on the 15-min grid: hourly values held over their 4 quarter-hours,
    gaps <= MAX_INTERP_H interpolated (hold=True: carried forward).
    Returns (values, gap report)."""
    s = s.dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    # each value covers its native step [t, t + step), step = smaller distance to a
    # neighbour (robust to a gap on one side), at most 1 h: hourly prices before
    # 10/2025 become piecewise constant on the quarter-hours
    t, q = s.index.asi8, qidx.asi8
    gaps = np.diff(t)
    step = np.minimum(np.r_[gaps, gaps[-1:]], np.r_[gaps[:1], gaps]).clip(max=3_600_000_000_000)
    i = np.searchsorted(t, q, side="right") - 1
    vals = np.full(len(q), np.nan)
    ok = i >= 0
    j = i[ok]
    cover = q[ok] < t[j] + step[j]
    vals[np.flatnonzero(ok)[cover]] = s.values[j[cover]]
    native = pd.Series(vals, index=qidx)
    missing = native.isna()
    runs = _runs(missing.values)
    lim = int(MAX_INTERP_H / timegrid.STEP_H)
    filled = native.ffill(limit=lim) if hold else native.interpolate(limit=lim, limit_area="inside")
    rep = {"missing_slots": int(missing.sum()), "n_gaps": len(runs),
           "longest_gap_h": max((b - a) * timegrid.STEP_H for a, b in runs) if runs else 0.0,
           "gaps": [[str(qidx[a]), (b - a) * timegrid.STEP_H] for a, b in runs[:50]],
           "unfilled_slots": int(filled.isna().sum())}
    return filled.values, rep


def _runs(mask: np.ndarray) -> list:
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(int))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def load(cache: str, qidx: pd.DatetimeIndex) -> tuple:
    """Driver frame on the 15-min grid qidx and the gap report.
    Columns: forecast_mw, actual_mw, price, installed_mw, cf_da, r_m."""
    out = pd.DataFrame(index=qidx)
    report = {}
    for series in SERIES:
        path = os.path.join(cache, f"{series}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"driver cache {path} missing - run the curt_drivers stage")
        s = pd.read_parquet(path)[series]
        out[series], report[series] = _on_grid(s, qidx, hold=(series == "price"))
        if report[series]["unfilled_slots"]:
            raise ValueError(f"{series}: {report[series]['unfilled_slots']} slots without data after gap "
                             f"filling (longest gap {report[series]['longest_gap_h']} h) - see the gap report")
    out["installed_mw"] = installed_mw(cache, qidx).values
    out["cf_da"] = out["forecast_mw"] / out["installed_mw"]
    out["r_m"] = monthly_market_value(cache).reindex(_months(qidx)).values
    if out["r_m"].isna().any():
        raise ValueError("market value r_m missing for some months - extend the driver cache")
    return out, report


def monthly_market_value(cache: str) -> pd.Series:
    """r_m from the whole cached span (full calendar months, also those only
    partly inside the synthesis period)."""
    pr = pd.read_parquet(os.path.join(cache, "price.parquet"))["price"]
    ac = pd.read_parquet(os.path.join(cache, "actual_mw.parquet"))["actual_mw"]
    lo, hi = max(pr.index.min(), ac.index.min()), min(pr.index.max(), ac.index.max())
    q = pd.date_range(lo.ceil("h"), hi.floor("h") - timegrid.STEP, freq=timegrid.STEP, tz="UTC")
    p, _ = _on_grid(pr, q, hold=True)
    a, _ = _on_grid(ac, q, hold=False)
    return market_value(pd.Series(p, index=q), pd.Series(a, index=q))


def _months(qidx: pd.DatetimeIndex) -> pd.PeriodIndex:
    return qidx.tz_convert(timegrid.TZ_LOCAL).tz_localize(None).to_period("M")


def market_value(price: pd.Series, actual: pd.Series) -> pd.Series:
    """Monthly onshore market value [EUR/MWh], months in Europe/Berlin."""
    m = _months(price.index)
    num = (price * actual).groupby(m).sum()
    den = actual.groupby(m).sum()
    return (num / den).rename("r_m")


def compare_local(drv: pd.DataFrame, prices_csv: str, wind_csv: str) -> dict:
    """Agreement of the API series with the local files of FL_Contribution/data/prices."""
    res = {}
    if os.path.exists(prices_csv):
        p = pd.read_csv(prices_csv, index_col=0)["price"]
        p.index = pd.to_datetime(p.index, utc=True)
        common = p.index.intersection(drv.index)
        d = (drv.loc[common, "price"] - p.loc[common]).abs()
        res["price"] = {"n": len(common), "max_abs_diff": float(d.max()), "n_diff_gt_0_01": int((d > 0.01).sum()),
                        "first": str(common.min()), "last": str(common.max())}
    if os.path.exists(wind_csv):
        w = pd.read_csv(wind_csv, index_col=0)
        w.index = pd.to_datetime(w.index, utc=True)
        common = w.index.intersection(drv.index)
        for a, b in (("forecast_mw", "forecast_da_mw"), ("actual_mw", "actual_mw")):
            d = (drv.loc[common, a] - w.loc[common, b]).abs()
            res[a] = {"n": len(common), "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
                      "rel_diff_energy": float((drv.loc[common, a].sum() - w.loc[common, b].sum())
                                               / w.loc[common, b].sum()),
                      "first": str(common.min()), "last": str(common.max())}
    return res
