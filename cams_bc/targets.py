"""Target forms and their inverse. A form is a pair (encode, decode) plus the sample weight that makes
the training loss an error in W/m² where the form is a ratio: for y = ghi/clear, sum w (y_hat - y)^2
with w = clear^2 equals sum (ghi_hat - ghi)^2."""
import numpy as np
import pandas as pd

FORMS = ("kc", "kc_unweighted", "ratio", "residual", "absolute")
_REF = {"ghi": "cams_ghi", "dhi": "cams_dhi"}


def encode(form: str, df: pd.DataFrame, var: str) -> tuple:
    """(y, weight) for target `var` in {'ghi', 'dhi'}."""
    obs = df[f"{var}_obs"].to_numpy(np.float64)
    clear = np.maximum(df["cams_ghi_clear"].to_numpy(np.float64), 1.0)
    ref = df[_REF[var]].to_numpy(np.float64)
    if form in ("kc", "kc_unweighted"):
        y = obs / clear
        w = clear ** 2 if form == "kc" else np.ones_like(y)
    elif form == "ratio":
        den = np.maximum(ref, 10.0)
        y, w = obs / den, den ** 2
    elif form == "residual":
        y, w = obs - ref, np.ones_like(obs)
    elif form == "absolute":
        y, w = obs, np.ones_like(obs)
    else:
        raise ValueError(form)
    return y, w / w.mean()


def decode(form: str, y_hat: np.ndarray, df: pd.DataFrame, var: str) -> np.ndarray:
    clear = np.maximum(df["cams_ghi_clear"].to_numpy(np.float64), 1.0)
    ref = df[_REF[var]].to_numpy(np.float64)
    if form in ("kc", "kc_unweighted"):
        out = y_hat * clear
    elif form == "ratio":
        out = y_hat * np.maximum(ref, 10.0)
    elif form == "residual":
        out = y_hat + ref
    else:
        out = y_hat
    return np.clip(out, 0.0, None)
