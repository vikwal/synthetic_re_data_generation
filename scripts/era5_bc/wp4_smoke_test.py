#!/usr/bin/env python3
"""WP4 smoke test — window integrity, model shapes, tiny overfit.

Run with the frcst venv:
    frcst/bin/python scripts/era5_bc/wp4_smoke_test.py
"""

import os
import sys

import numpy as np
import pandas as pd
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from era5_bc import static as S  # noqa: E402
from era5_bc.config import load_config  # noqa: E402
from era5_bc.data import load_processed  # noqa: E402
from era5_bc.models import build_model  # noqa: E402
from era5_bc.train import masked_huber  # noqa: E402
from era5_bc.windows import build_windows  # noqa: E402


def main():
    cfg = load_config()
    st, per = cfg["stations"], cfg["periods"]
    ids = st["train_effective"][:5]

    table = S.load_static_table(cfg)
    static_scaled = S.minmax_apply(table, S.minmax_fit(table, st["train_effective"]))

    start = per["train_start"]
    end = start + pd.Timedelta(days=40)
    ws = build_windows(cfg, ids, static_scaled, start, end)
    plen, seq = cfg["windows"]["past_len"], cfg["windows"]["past_len"] + 24

    # ---- shapes + ranges ----
    assert ws.Xd.shape[1:] == (seq, len(cfg["dynamic_covariates"])), ws.Xd.shape
    assert ws.Xs.shape[1] == len(cfg["static_covariates"])
    assert ws.Xdate.shape[1:] == (seq, 3)
    assert ws.y.shape[1] == 24 and ws.mask.shape[1] == 24
    assert np.isfinite(ws.Xd).all() and np.isfinite(ws.y).all()
    assert ws.Xdate[..., 2].min() >= 0 and ws.Xdate[..., 2].max() <= 23
    # target-day starts at local midnight: UTC hour == 24 - offset
    off = cfg["windows"]["timezone_offset_hours"]
    assert (pd.DatetimeIndex(ws.day_start).hour == (24 - off) % 24).all()
    print(f"windows OK: {len(ws)} samples from {len(ids)} stations")

    # ---- window content spot check against the processed frame ----
    i = len(ws) // 2
    sid, day = ws.station_ids[i], pd.Timestamp(ws.day_start[i], tz="UTC")
    frame = load_processed(cfg, sid)
    pos = frame.index.get_loc(day)
    y_expected = frame["y"].iloc[pos:pos + 24].to_numpy()
    valid = frame["y_valid"].iloc[pos:pos + 24].to_numpy()
    lo, hi = cfg["target"]["factor_clip"]
    assert np.allclose(ws.y[i][valid], np.clip(y_expected[valid], lo, hi),
                       atol=1e-6)
    assert (ws.mask[i] == valid.astype(np.float32)).all()
    # dynamic seq ends exactly at the last target hour (ws10 standardized)
    print(f"content OK ({sid} @ {day:%Y-%m-%d})")

    # ---- model shapes + tiny overfit ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sub = slice(0, 64)
    xd = torch.from_numpy(ws.Xd[sub]).to(device)
    xs = torch.from_numpy(ws.Xs[sub]).to(device)
    xdate = torch.from_numpy(ws.Xdate[sub]).to(device)
    y = torch.from_numpy(ws.y[sub]).to(device)
    mask = torch.from_numpy(ws.mask[sub]).to(device)

    for core in ("lstm", "transformer"):
        torch.manual_seed(0)
        model = build_model(cfg, core).to(device)
        out = model(xd, xs, xdate)
        assert out.shape == (64, 24) and (out >= 0).all()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        first = None
        for step in range(300):
            loss = masked_huber(model(xd, xs, xdate), y, mask,
                                cfg["model_common"]["huber_delta"])
            first = first or loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"{core}: params={model.count_parameters():,} "
              f"overfit {first:.4f} -> {loss.item():.4f}")
        assert loss.item() < 0.3 * first, f"{core} failed to overfit"

    print("smoke test PASSED")


if __name__ == "__main__":
    main()
