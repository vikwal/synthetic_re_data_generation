#!/usr/bin/env python
"""Synthetic PV park release: one run config, the stages of parks_pv.stages.

    synthre/bin/python generate_pv.py --config configs/runs/parks_pv_v4.yaml                  # all stages of the config
    synthre/bin/python generate_pv.py --config configs/runs/parks_pv_v4.yaml --stages generate,assemble --parks SEL1,SEL2

Locations outside the code come from settings.py (configs/settings.yaml or SYNTHRE_SETTINGS, environment overrides);
the run config holds what the release is (parks, period, step, model switches). See parks_pv/run_config.py.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parks_pv import run_config  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="run config YAML (configs/runs/)")
    ap.add_argument("--stages", default=None, help=f"comma list of {','.join(run_config.STAGES)} (default: stages of the config)")
    ap.add_argument("--parks", default=None, help="comma list of park ids (default: the whole selection)")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(argv)
    rc = run_config.load(a.config)
    if a.workers:
        rc["workers"] = a.workers
    stages = a.stages.split(",") if a.stages else rc["stages"]
    bad = set(stages) - set(run_config.STAGES)
    if bad:
        ap.error(f"unknown stages {sorted(bad)}")
    run_config.apply(rc)
    from parks_pv import paths, stages as S
    parks = a.parks.split(",") if a.parks else None
    print(f"release {paths.NAME}{paths.VARIANT}: selection {paths.SELECTION}, data {paths.DATA_DIR}, out {paths.RELEASE_DIR}", flush=True)
    for st in [s for s in run_config.STAGES if s in stages]:      # always in pipeline order
        t0 = time.time()
        print(f"=== {st}", flush=True)
        S.STAGE_FUNCS[st](rc, parks)
        print(f"=== {st} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
