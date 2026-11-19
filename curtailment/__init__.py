"""Curtailment as a switchable layer on top of the synthetic wind power.

Layers in settlement order (BNetzA BilAReM 2026, BDEW 2020):
  P_avail -> environment (bat curtailment) -> market (negative prices)
  -> grid (redispatch setpoints, P_obs = min(P_mkt, s * P_inst)).
Specification: FL_Contribution/reports/curtailment_modelling_review.md, section 6.
Entry points: generate_wind.apply_curtailment (single frame) and the stages
curt_* of scripts/parks/run_parks_v1.py (release of the 90 parks).
"""
