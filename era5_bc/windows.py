"""Day-window sample construction for the TR models.

One sample per (station, local day), following TRWindBC databuilder:
dynamic sequence = past_len context hours + the 24 target-day hours
(seq_len = past_len + 24), target = 24 hourly scaling factors.

Day boundaries and the temporal-embedding features use fixed CET (UTC+1,
no DST; config windows.timezone_offset_hours) — the reference implementation
uses station-local time. Context hours may extend before the period start:
they are ERA5-only covariates (no observations), so this is leakage-free.

Deviation from the paper: their ECCC data contains only complete 24 h days;
DWD hourly data has gaps, so invalid target hours are masked out of the loss
(mask weight 0) instead of dropping the day. Training targets are clipped to
target.factor_clip.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from . import data as D


class WindowSet:
    """Tensors + meta for one station set / target period.

    Attributes:
        Xd    [N, seq_len, n_dyn] normalized dynamic covariates
        Xs    [N, n_static]       min-max-scaled static covariates
        Xdate [N, seq_len, 3]     (month, day, hour) in fixed local time
        y     [N, 24]             scaling-factor targets (clipped, invalid -> 1.0)
        mask  [N, 24]             1.0 where the target hour is valid
        station_ids [N]           str array
        day_start   [N]           UTC timestamp of the first target hour
    """

    def __init__(self, Xd, Xs, Xdate, y, mask, station_ids, day_start):
        self.Xd, self.Xs, self.Xdate = Xd, Xs, Xdate
        self.y, self.mask = y, mask
        self.station_ids, self.day_start = station_ids, day_start

    def __len__(self):
        return len(self.y)

    def tensor_dataset(self) -> TensorDataset:
        return TensorDataset(
            torch.from_numpy(self.Xd), torch.from_numpy(self.Xs),
            torch.from_numpy(self.Xdate), torch.from_numpy(self.y),
            torch.from_numpy(self.mask))


def build_windows(cfg: dict, station_ids: list[str], static_scaled: pd.DataFrame,
                  period_start: pd.Timestamp, period_end: pd.Timestamp,
                  past_len: int | None = None, min_coverage: float | None = None,
                  clip_targets: bool = True,
                  require_targets: bool = True,
                  frame_ids: dict | None = None) -> WindowSet:
    """Build all valid day-windows for the stations in the given period.

    require_targets=False (prediction mode, e.g. park downscaling): every day
    with full ERA5 context is kept regardless of observations; y is a dummy
    and mask reflects obs validity (may be all-zero).
    frame_ids: optional mapping list-id -> processed-frame/norm-stats id.
    Used for parks: the park id selects the static covariates while the ERA5
    dynamics come from the park's name-giving DWD station (round2 chain
    convention)."""
    win, tgt = cfg["windows"], cfg["target"]
    plen = int(past_len if past_len is not None else win["past_len"])
    offset = int(win["timezone_offset_hours"])
    seq = plen + win["future_len"]
    min_valid = int(tgt["min_valid_hours_per_day"])
    norm_stats = D.load_norm_stats(cfg)

    parts = {k: [] for k in ("Xd", "Xdate", "y", "mask", "sid", "day")}
    skipped = []
    for sid in station_ids:
        fid = (frame_ids or {}).get(sid, sid)
        frame = D.load_processed(cfg, fid)
        if min_coverage is not None:
            if D.obs_coverage(frame, period_start, period_end) < min_coverage:
                skipped.append(sid)
                continue
        dyn = D.normalized_dynamics(frame, norm_stats[fid], cfg)  # [T, n_dyn]
        idx = frame.index
        local = idx + pd.Timedelta(hours=offset)
        dates = np.stack([local.month, local.day, local.hour], axis=-1).astype(np.int64)
        y_raw = frame["y"].to_numpy(np.float32)
        valid = frame["y_valid"].to_numpy(bool)

        # positions of local midnights with full context + full target day
        p = np.flatnonzero((local.hour == 0)
                           & (np.arange(len(idx)) >= plen)
                           & (np.arange(len(idx)) + 24 <= len(idx)))
        # target day fully inside the requested period
        p = p[(idx[p] >= period_start) & (idx[p] + pd.Timedelta(hours=23) <= period_end)]
        if len(p) == 0:
            continue
        if require_targets:
            tgt_idx = p[:, None] + np.arange(24)[None, :]
            n_valid = valid[tgt_idx].sum(axis=1)
            p = p[n_valid >= min_valid]
            if len(p) == 0:
                continue

        seq_idx = p[:, None] + np.arange(-plen, 24)[None, :]
        tgt_idx = p[:, None] + np.arange(24)[None, :]
        m = valid[tgt_idx]
        y = y_raw[tgt_idx]
        if clip_targets and tgt["factor_clip"] is not None:
            lo, hi = tgt["factor_clip"]
            y = np.clip(y, lo, hi)
        y = np.where(m, y, 1.0).astype(np.float32)

        parts["Xd"].append(dyn[seq_idx])
        parts["Xdate"].append(dates[seq_idx])
        parts["y"].append(y)
        parts["mask"].append(m.astype(np.float32))
        parts["sid"].append(np.repeat(sid, len(p)))
        parts["day"].append(idx[p].values)

    if skipped:
        print(f"build_windows: skipped {len(skipped)} stations below "
              f"min coverage {min_coverage}: {skipped}")
    assert parts["Xd"], "no windows built"

    sids = np.concatenate(parts["sid"])
    Xs = static_scaled.loc[sids, cfg["static_covariates"]].to_numpy(np.float32)
    return WindowSet(
        Xd=np.concatenate(parts["Xd"]).astype(np.float32),
        Xs=Xs,
        Xdate=np.concatenate(parts["Xdate"]),
        y=np.concatenate(parts["y"]),
        mask=np.concatenate(parts["mask"]),
        station_ids=sids,
        day_start=np.concatenate(parts["day"]),
    )
