"""Feature groups. Column names are produced here and nowhere else, so the dataset builder and the
model scripts cannot drift apart."""
from .config import AZIMUTHS, RADII

CAMS = ["cams_ghi", "cams_bhi", "cams_dhi", "cams_bni", "cams_ghi_clear", "cams_dhi_clear", "cams_bhi_clear",
        "cams_toa", "cams_reliability", "kc_cams", "kd_cams", "kt_cams", "fdir_cams"]
GEOM = ["cos_zen", "airmass", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
ERA5_SOLAR = ["era5_ghi", "era5_bhi", "era5_ghi_clear", "era5_toa", "era5_albedo", "era5_cloud_total", "era5_cloud_low",
              "era5_cloud_mid", "era5_cloud_high", "era5_water_vapour", "era5_k_c", "era5_f_dir", "era5_lw_cre"]
ERA5_MET = ["era5_temp_2m", "era5_pressure", "era5_dew_dep"]
STATIC = ["lat", "lon", "elev"]
N_NEIGHBOURS = 4
# Measured kc/kd and CAMS error of the k nearest other stations plus distance-weighted means. The
# per-neighbour distances are NOT features (decision 2026-10-13 after the review: constant per
# station, they act as a station identifier and cost 0.35 Pp objective); they only weight the means.
NEIGHBOURS = [f"nb{i}_{v}" for i in range(N_NEIGHBOURS) for v in ("kc", "kd", "err")] + ["nb_err_wmean", "nb_kc_wmean", "nb_err_std"]


def ring_cols(radius: float) -> list:
    return [f"ring{radius:g}{a}_{v}" for a in AZIMUTHS for v in ("kc", "kd")]


RING3 = ring_cols(3.0)
RING8 = ring_cols(8.0)
RING_AGG = [f"ring{r:g}_kc_{s}" for r in RADII for s in ("mean", "std", "ns", "ew")] + \
           [f"ring{r:g}_kd_mean" for r in RADII] + ["ring_all_kc_mean", "centre_minus_ring3"]
CONTEXT = ["kc_lag_m1", "kc_lag_p1", "kc_lag_m2", "kc_lag_p2", "kc_std5", "kc_mean5", "kc_daymean", "kc_daystd",
           "ring3_kc_lag_m1", "ring3_kc_lag_p1", "ring8_kc_lag_m1", "ring8_kc_lag_p1", "kd_lag_m1", "kd_lag_p1"]

GROUPS = {"cams": CAMS, "geom": GEOM, "ring3": RING3, "ring8": RING8, "ring_agg": RING_AGG, "context": CONTEXT,
          "era5_solar": ERA5_SOLAR, "era5_met": ERA5_MET, "neighbours": NEIGHBOURS, "static": STATIC}

# cumulative ladders used by the predictor study
LADDER = [["cams", "geom"], ["cams", "geom", "ring3"], ["cams", "geom", "ring3", "ring8"],
          ["cams", "geom", "ring3", "ring8", "ring_agg"], ["cams", "geom", "ring3", "ring8", "ring_agg", "context"],
          ["cams", "geom", "ring3", "ring8", "ring_agg", "context", "era5_solar"],
          ["cams", "geom", "ring3", "ring8", "ring_agg", "context", "era5_solar", "era5_met"],
          ["cams", "geom", "ring3", "ring8", "ring_agg", "context", "era5_solar", "era5_met", "neighbours"],
          ["cams", "geom", "ring3", "ring8", "ring_agg", "context", "era5_solar", "era5_met", "neighbours", "static"]]
FULL = LADDER[-1]
BEST = ["cams", "geom", "ring3", "ring8", "ring_agg", "context", "era5_solar", "era5_met", "neighbours"]
TARGET_COLS = ["ghi_obs", "dhi_obs", "y_valid_ghi", "y_valid_dhi", "day_mask", "src_ghi", "src_dhi"]


def columns(groups) -> list:
    out = []
    for g in groups:
        out += GROUPS[g]
    return out
