"""Anchor regression test: generate_wind_era5_v2 with all round-2 switches
off must reproduce generate_wind_era5 (round 1) bit-for-bit on one park.

Runs both mains in subprocesses (their argparse would clash with pytest's
argv) against a scratch synth_dir, on the existing ERA5 CSVs.
"""

import os
import shutil
import subprocess
import sys

import pandas as pd
import pytest
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON = os.path.join(REPO, "synthre", "bin", "python")
PARK = "07374"
SCRATCH = "/mnt/nvme2/synthetic/tmp_regression_test"


def _write_config(name: str, with_round2: bool):
    src = os.path.join(REPO, "configs", "real_wind_parks_era5", f"config_{PARK}.yaml")
    with open(src) as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["synth_dir"] = SCRATCH
    if with_round2:
        cfg["round2"] = {
            "experiment_id": "regtest",
            "correction": "off",
            "shear": "power_law",
            "aging_model": "const",
            "wake": {"enabled": False},
            "density": "v1_mixed",
            "era5_dir_v2": None,
        }
    dst = os.path.join(REPO, "configs", name)
    with open(dst, "w") as f:
        yaml.safe_dump(cfg, f)
    return dst


def _run(module: str, config_name: str):
    code = (f"import sys; sys.argv=['x']; "
            f"import {module} as m; m.main('{config_name}')")
    res = subprocess.run([PYTHON, "-c", code], cwd=REPO,
                         capture_output=True, text=True, timeout=3600)
    assert res.returncode == 0, f"{module} failed:\n{res.stderr[-3000:]}"


@pytest.mark.slow
def test_v2_all_off_equals_v1():
    shutil.rmtree(SCRATCH, ignore_errors=True)
    os.makedirs(SCRATCH, exist_ok=True)
    cfgs = []
    try:
        cfgs.append(_write_config(f"config_{PARK}.yaml", with_round2=False))
        _run("generate_wind_era5", f"config_{PARK}.yaml")
        v1_path = os.path.join(SCRATCH, "wind", "era5_wind_hourly_age",
                               f"synth_{PARK}.csv")
        v1 = pd.read_csv(v1_path, sep=";", index_col=0, parse_dates=True)

        for c in cfgs:
            os.remove(c)
        cfgs = [_write_config(f"config_{PARK}.yaml", with_round2=True)]
        _run("generate_wind_era5_v2", f"config_{PARK}.yaml")
        v2_path = os.path.join(SCRATCH, "wind", "round2", "regtest",
                               f"synth_{PARK}.csv")
        v2 = pd.read_csv(v2_path, sep=";", index_col=0, parse_dates=True)

        power_cols = [c for c in v1.columns if c.startswith("power_t")]
        assert power_cols, f"no power columns in v1 output: {v1.columns.tolist()}"
        for col in power_cols:
            pd.testing.assert_series_equal(v1[col], v2[col], rtol=0, atol=1e-9,
                                           check_exact=False)
        # v2 extras must be consistent
        assert "power_park" in v2.columns
        pd.testing.assert_series_equal(
            v2["power_park"], v2[power_cols].sum(axis=1).rename("power_park"),
            rtol=0, atol=1e-6)
    finally:
        for c in cfgs:
            if os.path.exists(c):
                os.remove(c)
