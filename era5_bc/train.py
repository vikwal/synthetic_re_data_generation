"""Training loop for the TR models (reference: TRWindBC train_predict.py).

Huber loss (delta 1.5), Adam (weight_decay 0), batch 128. Deviations from the
reference: masked loss (invalid target hours get weight 0; their data has only
complete days) and early stopping with patience (they train a fixed 30 epochs
and pick the min-val-loss checkpoint — we keep the checkpoint selection and
just stop earlier when the val loss has stopped improving).
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .models import TRModel
from .windows import WindowSet


def masked_huber(pred, target, mask, delta):
    loss = torch.nn.functional.huber_loss(pred, target, reduction="none",
                                          delta=delta)
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def make_loader(ws: WindowSet, cfg: dict, shuffle: bool,
                batch_size: int | None = None) -> DataLoader:
    mc = cfg["model_common"]
    return DataLoader(ws.tensor_dataset(),
                      batch_size=int(batch_size or mc["batch_size"]),
                      shuffle=shuffle, drop_last=shuffle,
                      num_workers=int(mc["num_workers"]), pin_memory=True,
                      persistent_workers=int(mc["num_workers"]) > 0)


@torch.no_grad()
def station_median_loss(model, loader, station_ids, device, delta):
    """Median over stations of the per-station mean Huber loss.

    Robust early-stopping monitor for the final runs: a pooled mean over few
    monitor stations is dominated by a single extreme-factor station (e.g. a
    summit like Feldberg with median y~3.6), which stalls the monitor and
    stops training far too early."""
    model.eval()
    loss_sums, mask_sums = [], []
    for xd, xs, xdate, y, mask in loader:
        pred = model(xd.to(device), xs.to(device), xdate.to(device))
        loss = torch.nn.functional.huber_loss(
            pred, y.to(device), reduction="none", delta=delta)
        loss_sums.append((loss * mask.to(device)).sum(dim=1).cpu())
        mask_sums.append(mask.sum(dim=1))
    df = pd.DataFrame({"sid": station_ids,
                       "l": torch.cat(loss_sums).numpy(),
                       "n": torch.cat(mask_sums).numpy()})
    per_station = df.groupby("sid").sum()
    return float((per_station["l"] / per_station["n"].clip(lower=1)).median())


def run_epoch(model, loader, device, delta, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, n = 0.0, 0
    with torch.set_grad_enabled(training):
        for xd, xs, xdate, y, mask in loader:
            xd, xs, xdate = xd.to(device), xs.to(device), xdate.to(device)
            y, mask = y.to(device), mask.to(device)
            pred = model(xd, xs, xdate)
            loss = masked_huber(pred, y, mask, delta)
            if training:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            total += loss.item()
            n += 1
    return total / max(n, 1)


def fit(model: TRModel, train_ws: WindowSet, val_ws: WindowSet, cfg: dict,
        lr: float, ckpt_dir: str, device: str = "cuda",
        weight_decay: float = 0.0, batch_size: int | None = None,
        max_epochs: int | None = None, epoch_callback=None,
        monitor: str = "pooled", patience: int | None = None) -> pd.DataFrame:
    """Train with early stopping; keeps the best (min val loss) state dict at
    {ckpt_dir}/best.pt and the log at {ckpt_dir}/log.csv. Returns the log."""
    mc = cfg["model_common"]
    torch.manual_seed(int(mc["seed"]))
    np.random.seed(int(mc["seed"]))
    max_epochs = int(max_epochs or mc["max_epochs"])
    patience = int(patience if patience is not None
                   else mc["early_stopping_patience"])
    delta = float(mc["huber_delta"])

    os.makedirs(ckpt_dir, exist_ok=True)
    train_loader = make_loader(train_ws, cfg, shuffle=True, batch_size=batch_size)
    val_loader = make_loader(val_ws, cfg, shuffle=False, batch_size=batch_size)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    print(f"parameters: {model.count_parameters():,} | train windows: "
          f"{len(train_ws):,} | val windows: {len(val_ws):,}")

    log, best_vloss, best_epoch = [], float("inf"), -1
    assert monitor in ("pooled", "station_median")
    for epoch in range(max_epochs):
        tloss = run_epoch(model, train_loader, device, delta, optimizer)
        if monitor == "station_median":
            vloss = station_median_loss(model, val_loader, val_ws.station_ids,
                                        device, delta)
        else:
            vloss = run_epoch(model, val_loader, device, delta)
        log.append({"epoch": epoch, "lr": lr, "tloss": tloss, "vloss": vloss})
        marker = ""
        if vloss < best_vloss:
            best_vloss, best_epoch = vloss, epoch
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pt"))
            marker = " *"
        print(f"epoch {epoch:03d} | train {tloss:.4f} | val {vloss:.4f}{marker}",
              flush=True)
        pd.DataFrame(log).to_csv(os.path.join(ckpt_dir, "log.csv"), index=False)
        if epoch_callback is not None:
            epoch_callback(epoch, tloss, vloss)
        if epoch - best_epoch >= patience:
            print(f"early stop at epoch {epoch} (best {best_epoch}, "
                  f"vloss {best_vloss:.4f})")
            break
    return pd.DataFrame(log)


@torch.no_grad()
def predict(model: TRModel, ws: WindowSet, cfg: dict,
            device: str = "cuda") -> pd.DataFrame:
    """Predict scaling factors; returns a long frame
    (station_id, timestamp UTC, sf_pred) with one row per target hour."""
    model = model.to(device).eval()
    loader = make_loader(ws, cfg, shuffle=False)
    preds = [model(xd.to(device), xs.to(device), xdate.to(device)).cpu().numpy()
             for xd, xs, xdate, _, _ in loader]
    sf = np.concatenate(preds)  # [N, 24]

    hours = np.arange(24)
    ts = (ws.day_start[:, None] + hours[None, :] * np.timedelta64(1, "h")).ravel()
    return pd.DataFrame({
        "station_id": np.repeat(ws.station_ids, 24),
        "timestamp": pd.DatetimeIndex(ts, tz="UTC"),
        "sf_pred": sf.ravel(),
    })
