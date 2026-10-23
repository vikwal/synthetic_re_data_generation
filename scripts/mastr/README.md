# MaStR processing for the wind fleet (and early PV checks)

Copied on 2026-10-23 from `~/Work/FL_Contribution/pipeline/mastr/` (no git there) so that the register processing lives
with the generator. The scripts are unchanged; inputs and outputs stay under `~/Work/FL_Contribution/data/mastr/`
(paths in the scripts). The PV counterparts were already here: `pv/mastr.py`, `scripts/pv/mastr_parse.py`,
`scripts/pv/select_parks.py`, `scripts/pv/replace_parks.py` (selection v2), `scripts/pv/trianel_mastr_match.py`,
and the PV transfer split `scripts/pv/pv_transfer_split.py` (copied from `FL_Contribution/pipeline/`).

| Script | Purpose |
|---|---|
| `wind_check.py` | MaStR wind units near the synthetic sites, turbine-type match against the power-curve library -> `wind_turbines_matched.csv`, `wind_candidate_parks.csv` |
| `select_real_parks.py` | first wind selection (8 regional clients x 10-park ladders + 10 held out) |
| `select_real_parks_v1.py` | wind selection v1 (composition classes, cohorts) |
| `select_real_parks_v1_1.py` | v1.1: replaces parks whose MaStR coordinates disagree with OSM (uses `osm_match.py`) |
| `osm_match.py` | MaStR turbine coordinates vs OSM wind generators (Hungarian assignment, ok / corrected / flagged) |
| `park_composition_stats.py` | composition statistics of the wind parks (types, sizes, cohorts) |
| `build_park_map.py` + `park_map*_template.html` | HTML map of the selected parks (output not tracked) |
| `solar_check.py`, `solar_xml_sample.py` | early PV register checks (sample of the MaStR solar export) |
