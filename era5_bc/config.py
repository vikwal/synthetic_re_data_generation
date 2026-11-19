"""Config loading + validation for the ERA5 bias-correction experiment.

Reimplementation of Houndekindo & Ouarda (2025), Energy 328, 136498.
Hard leakage rule (round2-conform, see scripts/round2/wp2a_station_table.py):
the training period must never overlap the evaluation window
(round2 validation window Jun 2023 - Jun 2024).
"""

import os

import pandas as pd
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CONFIG = os.path.join(REPO, "configs", "config_era5_bc.yaml")

SPLIT_SIZES = {"train": 103, "val": 50, "test": 50}


def load_config(path: str = DEFAULT_CONFIG) -> dict:
    """Load and validate the experiment config. Returns the config dict with
    resolved (repo-absolute) paths and effective station splits."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # --- resolve relative paths against the repo root ---
    for key, val in cfg["paths"].items():
        if not os.path.isabs(val):
            cfg["paths"][key] = os.path.join(REPO, val)

    # --- station split sanity ---
    st = cfg["stations"]
    for split, n in SPLIT_SIZES.items():
        assert len(st[split]) == n, f"{split}: expected {n} ids, got {len(st[split])}"
        assert len(set(st[split])) == n, f"{split}: duplicate station ids"
        assert all(isinstance(s, str) and len(s) == 5 for s in st[split]), \
            f"{split}: ids must be 5-char zero-padded strings"
    all_ids = st["train"] + st["val"] + st["test"]
    assert len(set(all_ids)) == len(all_ids), "station splits overlap"

    # effective splits after dropping stations without ERA5 data
    missing = set(st.get("missing_era5", []))
    if st.get("drop_missing_era5", True):
        for split in SPLIT_SIZES:
            st[f"{split}_effective"] = [s for s in st[split] if s not in missing]
    else:
        for split in SPLIT_SIZES:
            st[f"{split}_effective"] = list(st[split])

    # --- period parsing + hard leakage assert ---
    per = cfg["periods"]
    for key in ("train_start", "train_end", "eval_start", "eval_end"):
        per[key] = pd.Timestamp(per[key], tz="UTC")
    assert per["train_start"] < per["train_end"]
    assert per["eval_start"] < per["eval_end"]
    assert per["eval_end"] < per["train_start"], (
        "leakage: training period overlaps the evaluation window "
        f"(eval_end={per['eval_end']}, train_start={per['train_start']})"
    )

    # --- window params ---
    win = cfg["windows"]
    assert win["future_len"] == 24, "future_len must be 24 (one target day per window)"
    assert win["past_len"] % 24 == 0, "past_len must be a multiple of 24"

    return cfg


def early_stop_split(cfg: dict, split: str) -> tuple[list[str], list[str]]:
    """Split the given eval split ('val' | 'test') into (early-stop stations,
    evaluation stations). Seeded, deterministic. The early-stop stations drive
    early stopping of the final training; the remaining stations stay untouched
    for evaluation."""
    import random

    assert split in ("val", "test")
    ids = list(cfg["stations"][f"{split}_effective"])
    n = int(cfg["stations"]["n_early_stop"])
    rng = random.Random(int(cfg["model_common"]["seed"]))
    es = sorted(rng.sample(ids, n))
    evaluate = [s for s in ids if s not in es]
    return es, evaluate


def station_era5_path(cfg: dict, station_id: str) -> str:
    return os.path.join(cfg["paths"]["era5_dir"], f"Station_{station_id}.csv")


def station_dwd_path(cfg: dict, station_id: str) -> str:
    return os.path.join(cfg["paths"]["dwd_dir"], f"DWD_{station_id}.parquet")


def station_processed_path(cfg: dict, station_id: str) -> str:
    return os.path.join(cfg["paths"]["processed_dir"], f"Station_{station_id}.parquet")
