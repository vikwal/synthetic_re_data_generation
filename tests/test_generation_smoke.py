"""Integration smoke test for generate_wind.main(): runs the generator
end-to-end on one station and checks the output is internally consistent
(park power is the sum of its turbines' power).

Marked slow: needs the real ERA5 input on disk and takes minutes to run.
"""

import os
import shutil
import subprocess

import pandas as pd
import pytest
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON = os.path.join(REPO, "synthre", "bin", "python")
PARK = "07374"
SCRATCH = "/mnt/nvme2/synthetic/tmp_regression_test"


def _write_config(name: str):
    src = os.path.join(REPO, "configs", "round2", "M1.yaml")
    with open(src) as f:
        cfg = yaml.safe_load(f)
    base_src = os.path.join(REPO, "configs", "config_wind.yaml")
    with open(base_src) as f:
        base_cfg = yaml.safe_load(f)
    base_cfg["data"]["synth_dir"] = SCRATCH
    base_cfg["round2"] = cfg["round2"]
    base_cfg["round2"]["experiment_id"] = "smoketest"
    dst = os.path.join(REPO, "configs", name)
    with open(dst, "w") as f:
        yaml.safe_dump(base_cfg, f)
    return dst


def _run(config_name: str):
    code = ("import sys; sys.argv=['x']; "
            f"import generate_wind as m; m.main('{config_name}')")
    res = subprocess.run([PYTHON, "-c", code], cwd=REPO,
                         capture_output=True, text=True, timeout=3600)
    assert res.returncode == 0, f"generate_wind failed:\n{res.stderr[-3000:]}"


@pytest.mark.slow
def test_generate_wind_smoke():
    shutil.rmtree(SCRATCH, ignore_errors=True)
    os.makedirs(SCRATCH, exist_ok=True)
    cfg_path = None
    try:
        cfg_path = _write_config(f"config_{PARK}.yaml")
        _run(f"config_{PARK}.yaml")
        out_path = os.path.join(SCRATCH, "wind", "round2", "smoketest",
                                f"synth_{PARK}.csv")
        df = pd.read_csv(out_path, sep=";", index_col=0, parse_dates=True)

        power_cols = [c for c in df.columns if c.startswith("power_t")]
        assert power_cols, f"no power columns in output: {df.columns.tolist()}"
        assert "power_park" in df.columns
        pd.testing.assert_series_equal(
            df["power_park"], df[power_cols].sum(axis=1).rename("power_park"),
            rtol=0, atol=1e-6)
    finally:
        if cfg_path and os.path.exists(cfg_path):
            os.remove(cfg_path)
