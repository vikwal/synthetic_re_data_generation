#!/usr/bin/env python
"""Rewrite the legacy real-park configs (configs/real_pv_parks/config_<name>.yaml, not runnable:
tilt_az_list nested one level too deep, DC in watts, no coordinates) into the pv.config schema,
one file per anonymised park id (pv1..pv7). The directory is gitignored (confidential)."""

import glob
import os
import sys

import pandas as pd
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pv import config as pvconfig  # noqa: E402
import trianel_common as tc  # noqa: E402


def main():
    m = tc.load_masterdata()
    parks = tc.parks_table(m)
    meter = tc.load_meter_hourly(parks)
    facts = tc.mastr_facts(parks, *(sys.argv[1:3] if len(sys.argv) >= 3 else (None, None)))
    par = tc.park_parameters(parks, facts, meter)
    out_dir = os.path.join(REPO, "configs", "real_pv_parks")
    for pv, p in parks.iterrows():
        cfg = tc.base_config(pv, p, par.loc[pv], variant="C")
        pvconfig.dump(cfg, os.path.join(out_dir, f"{pv}.yaml"))
        print(pv, "->", f"{pv}.yaml", f"{cfg['kwp_dc']:.0f} kWp, dc_ac {cfg['dc_ac']:.2f}, {len(cfg['arrays'])} arrays, module {cfg['module']['name']}")
    for f in glob.glob(os.path.join(out_dir, "config_*.yaml")):
        old = yaml.safe_load(open(f))
        try:
            c = pvconfig.from_legacy(old, os.path.basename(f)[7:-5], 0.0, 0.0)
            print("legacy", os.path.basename(f), "parsed:", [(a["tilt"], a["azimuth"]) for a in c["arrays"]], f"{c['kwp_dc']:.0f} kWp")
        except Exception as e:  # noqa: BLE001
            print("legacy", os.path.basename(f), "FAILED", e)


if __name__ == "__main__":
    main()
