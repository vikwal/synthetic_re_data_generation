# configs/round2/

Each YAML here is a full configuration for `generate_wind.py` (`--config round2/<name>.yaml`).
`M4.yaml` is the framework default (see repository root `README.md`).

## Naming

Filenames match the manuscript's configuration labels (Table
`tab:experimental_framework`): `M1`-`M4` is the deployed dynamic-power-law
path, `S1`-`S3` the validated stability-corrected alternative. `S4`
(height-gated hybrid) has no single config: it is a per-farm combination of
the `M4` and `S3` outputs, selected by whether the turbine's hub height is
above or below 100 m (see `round2/paper_style.py:PAPER_RUNG` for how each
config's internal `experiment_id` maps to its manuscript label, used
throughout `results/quantitative/`).

`*_noage` / `*_noQM` files are single-component leave-one-out ablations
(e.g. `M4_noage.yaml` = `M4.yaml` with aging disabled), used for the
component-attribution contrasts in the paper's results section, not
individually reported as their own rows.

Remaining files (`M2_trbc*`, `M3_vg`, `M4b*`, `M5_gate5`, `M5_hybgate`,
`M5_vg*`, `M5all_distgate`, etc.) are supplementary sensitivity/ablation
variants explored during development; their results feed the full CSVs in
`results/quantitative/` but are not each named individually in the paper text.

`_generated/` holds per-station merged configs produced by
`scripts/round2/run_ladder.py` (not hand-edited).
