"""WP1 — evaluation module (guide section WP1).

evaluate() computes all headline metrics for one park; compare_pathways()
runs the N=13 paired statistics (exact Wilcoxon, rank-biserial effect size,
sign test, park-level block bootstrap). Unit of observation = wind park.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf as sm_acf


def _align(p_meas: pd.Series, p_synth: pd.Series):
    df = pd.concat([p_meas.rename("meas"), p_synth.rename("synth")], axis=1).dropna()
    return df["meas"], df["synth"]


def evaluate(p_meas: pd.Series, p_synth: pd.Series, p_rated: float,
             prices: pd.Series = None, acf_lags: int = 48) -> dict:
    """All WP1 metrics for one park over the (already clipped) window.

    p_meas/p_synth: hourly park power in the SAME unit (W), UTC index.
    p_rated: park rated power (W) for normalization.
    prices: optional hourly day-ahead prices (EUR/MWh) for the curtailment
    screen; metrics are reported with and without price<=0 hours.
    """
    meas, synth = _align(p_meas, p_synth)
    out = _metrics(meas, synth, p_rated, acf_lags)
    out["n_hours"] = int(len(meas))
    out["window"] = (str(meas.index.min()), str(meas.index.max()))
    if prices is not None:
        flagged = prices.reindex(meas.index)["price_eur_mwh"] <= 0 \
            if isinstance(prices, pd.DataFrame) else prices.reindex(meas.index) <= 0
        flagged = flagged.fillna(False)
        out["curtailed_share"] = float(flagged.mean())
        sub = _metrics(meas[~flagged], synth[~flagged], p_rated, acf_lags)
        out.update({f"{k}_excl_curt": v for k, v in sub.items()})
    return out


def _metrics(meas: pd.Series, synth: pd.Series, p_rated: float, acf_lags: int) -> dict:
    if len(meas) < 10:
        return {"r2": np.nan, "rmse_n": np.nan, "mae_n": np.nan,
                "energy_ratio": np.nan, "wasserstein": np.nan}
    yn_t, yn_p = meas / p_rated, synth / p_rated
    ss_res = float(((meas - synth) ** 2).sum())
    ss_tot = float(((meas - meas.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    out = {
        "r2": r2,
        "rmse_n": float(np.sqrt(((yn_t - yn_p) ** 2).mean())),
        "mae_n": float((yn_t - yn_p).abs().mean()),
        "energy_ratio": float(synth.sum() / meas.sum()) if meas.sum() > 0 else np.nan,
        "wasserstein": float(stats.wasserstein_distance(yn_t, yn_p)),
    }
    try:
        out["acf_meas"] = sm_acf(meas, nlags=acf_lags, missing="drop").tolist()
        out["acf_synth"] = sm_acf(synth, nlags=acf_lags, missing="drop").tolist()
    except Exception:
        out["acf_meas"] = out["acf_synth"] = None
    return out


def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Matched-pairs rank-biserial r = 1 - 2*W_min / (n(n+1)/2), zeros dropped."""
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    w_plus = ranks[d > 0].sum()
    w_minus = ranks[d < 0].sum()
    w_min = min(w_plus, w_minus)
    r = 1.0 - 2.0 * w_min / (n * (n + 1) / 2.0)
    return float(r if w_plus >= w_minus else -r)


def park_turbine_counts() -> pd.Series:
    """Turbines per park from park_layouts.csv (index park_id).

    Used to restrict wake-component significance tests to multi-turbine
    parks: single-turbine parks have no intra-park wakes, so their paired
    difference is structurally zero and only dilutes the test."""
    import os
    layouts = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "..", "data", "round2",
                     "park_layouts.csv"),
        dtype={"park_id": str})
    return layouts.groupby("park_id").size().rename("n_turbines")


def compare_pathways(metric_a: pd.Series, metric_b: pd.Series,
                     n_boot: int = 10_000, seed: int = 42) -> dict:
    """Paired comparison of one metric across parks between two pathways.

    metric_a/metric_b: per-park values, aligned on park_id index (N = 13).
    Positive median_diff means pathway A > pathway B.
    """
    df = pd.concat([metric_a.rename("a"), metric_b.rename("b")], axis=1).dropna()
    x, y = df["a"].values, df["b"].values
    n = len(df)
    out = {"n_parks": n, "median_diff": float(np.median(x - y))}
    if n < 5 or np.allclose(x, y):
        out.update({"wilcoxon_p": np.nan, "rank_biserial": 0.0,
                    "sign_test_p": np.nan, "boot_ci_lo": np.nan, "boot_ci_hi": np.nan})
        return out
    w = stats.wilcoxon(x, y, method="exact")
    out["wilcoxon_stat"] = float(w.statistic)
    out["wilcoxon_p"] = float(w.pvalue)
    out["rank_biserial"] = rank_biserial(x, y)
    d = x - y
    nz = d[d != 0]
    k = int((nz > 0).sum())
    out["sign_test_p"] = float(stats.binomtest(k, len(nz), 0.5).pvalue) if len(nz) else np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = np.median(x[idx] - y[idx], axis=1)
    out["boot_ci_lo"], out["boot_ci_hi"] = map(float, np.percentile(boot, [2.5, 97.5]))
    return out
