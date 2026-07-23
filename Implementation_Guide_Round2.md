# Implementation Guide — Round-2 Revision of the Synthetic Wind Power Framework

**Audience:** whoever implements this next to the existing framework code (the pipeline that produces `era5`, `era5+aging`, `dwd+era5`, `dwd+era5+aging` from ERA5 + DWD inputs).
**Self-contained:** no other document is required. Background/reviewer mapping lives in `Revision_Plan_Round2.md` and `Methodology_Improvements_Round2.md` (paper repo), but everything needed to *build* is in here.
**Scope:** 8 work packages (WP0–WP7) + final experiment ladder. Target: all results for the Applied Energy round-2 resubmission.

---

## Overview & execution order

```
WP0  Data acquisition            (blocking for everything)
WP1  Evaluation module upgrade   (do FIRST after WP0 — every WP validates through it)
WP2  ERA5 classification + gated bias correction ("downscaling")
WP3  Stability-corrected extrapolation (MOST)
WP4  Aging v2 (Weibull wear-out + EEG-step test + fleet recalibration)
WP5  Wake losses (PyWake)
WP6  Global sensitivity analysis (Morris → Sobol)
WP7  Uncertainty bands (OPTIONAL / stretch — see triage note in WP7)
```

**Priority if time gets tight** (reviewer-demand order): WP1 → WP2 → WP4 → WP6 → WP5 → WP3 (at minimum its stability-stratified error table) → WP7 (delta-method paragraph only; MC = stretch or follow-up paper).

Dependencies: WP2–WP5 are independent of each other (parallelizable) once WP0+WP1 exist. WP6 needs all chain components switchable (M-ladder below). WP7 reuses WP6's parameter ranges. Final numbers for WP4's validation should be produced *after* WP2/WP3 are frozen (aging regression uses the corrected wind).

**Reproducibility ground rules:** pin all package versions (`requirements.txt`), fix random seeds, one YAML config per experiment, never overwrite raw downloads. Pin **PyWake's version in the paper text** (implementations of the same wake model differ across tools — TORQUE 2024 finding).

---

## WP0 — Data acquisition

| # | Dataset | Source | Details |
|---|---|---|---|
| 0.1 | ERA5 additions: `sshf` (surface sensible heat flux), `blh` (boundary-layer height), `gwd` (gravity-wave dissipation) | CDS API (same single-level API already in use) | Hourly, 2015-01 … 2024-06, at the 200 stations + 13 parks + 160 sites. `sshf` is **accumulated J/m², positive downward** → upward flux in W/m² is `H = −sshf/3600`. |
| 0.2 | ERA5 10 m / 100 m u,v for the *station* locations | CDS (already have for parks/sites) | 2015–2024, needed for the station correction table (WP2-A). |
| 0.3 | DWD 10 m wind, ~200 stations | DWD CDC OpenData (already partly in-house) | Hourly, 2015–2024. Train window 2015–2022; **never** use Jun 2023–Jun 2024 for fitting (reserved for park validation). |
| 0.4 | DEM | Copernicus DEM GLO-30 (AWS `copernicus-dem-30m` or OpenTopography) | Germany tiles. Aggregate to ~90 m/1 km for large-radius metrics. |
| 0.5 | Land cover → roughness z₀ | CORINE Land Cover 2018 (Copernicus Land, free login) | Reclassify to z₀ via standard lookup (e.g. Silva et al. 2007). |
| 0.6 | Turbine coordinates of the 13 parks | operator data (confirmed available) | Per-turbine x,y (+ model, hub height). |
| 0.7 | Thrust curves CT(v) for the 6+ turbine models | wind-turbine-models.com / WAsP .wtg / PyWake examples | **Prerequisite for WP5.** If a CT curve is missing, use PyWake's generic CT and flag as assumption. |
| 0.8 | Day-ahead prices (curtailment proxy) | ENTSO-E Transparency (free API) | DE-LU zone, Jun 2023–Jun 2024. Optional: netztransparenz.de Redispatch lists. |

**Derived topo metrics** (from 0.4, per station/park/site): elevation, slope, aspect, TPI @ 5 km & @ 75 km radius, TDI = (H_max−H_min)/H_mean in 11 km window, elevation-std (as already used in the paper), z₀ from 0.5. Tools: `richdem`, `WhiteboxTools` (`RelativeTopographicPosition` supports custom radius), or GDAL `gdaldem`. Use the coarse DEM for the 75 km radius (compute cost).

---

## WP1 — Evaluation module upgrade (do first)

Everything downstream is judged through this module; build it before touching the model.

**1.1 Metrics (add to the existing normalized-error + R²):**
- **Energy ratio** per park: `ER = Σ P_synth / Σ P_meas` over the validation window. The headline metric for a dataset meant for energy modelling.
- **Wasserstein-1 distance** between normalized power distributions: `scipy.stats.wasserstein_distance(P_meas/P_rated, P_synth/P_rated)`. Rationale: models tying on MAE can differ by ~15 pp in cumulative energy (Schmidt & Ludwig 2026); distribution shape is what the power curve cubes.
- **ACF comparison**: autocorrelation of measured vs synthetic power, lags 1–48 h (`statsmodels.tsa.stattools.acf`). One overlay figure per pathway (reviewer 2 explicitly asked for temporal correlations).
- Keep normalized error + R² for continuity with round 1.

**1.2 Statistics fix (mandatory, small):**
- Unit of observation = **wind park** (N = 13), state it explicitly.
- `scipy.stats.wilcoxon(x, y, method="exact")` — exact test, not normal approximation (N < 25).
- Report **effect size** alongside p: matched-pairs rank-biserial `r = 1 − 2·W_min / (n(n+1)/2)`, plus median paired difference in physical units (ΔR², Δ energy ratio).
- Robustness companion: sign test or park-level block bootstrap (resample parks, 10⁴ draws) for the key comparisons.

**1.3 Curtailment screen:**
- Flag hours with day-ahead price ≤ 0 €/MWh (proxy for feed-in management / market-driven curtailment).
- Report every headline metric **with and without** flagged hours (sensitivity, not exclusion by default). Optional upgrade: park-region Redispatch lists.

**Definition of done:** one `evaluate(park, synth_series)` function returning all metrics; regression-tested against the round-1 numbers (reproduce the published R² table within tolerance).

---

## WP2 — ERA5 quality classification + gated bias correction

### The height rule (read first — this makes or breaks the WP)

The hub-height wind is algebraically `v(h) = v10^(1−β) · v100^β`, `β = ln(h/10)/ln(10)` (β = 0.76 @ 57 m, 0.98 @ 95 m, 1.14 @ 138 m). Hence a correction applied **to v10 only** changes v(hub) by `(1+c)^(1−β)` → **nil at ~100 m, sign-inverted above**. Therefore:

> **Height-consistent rule:** apply the *relative* (quantile) correction factor to **both v10 and v100**. α is then unchanged and v(hub) scales exactly by (1+c). State the height-invariance assumption in the paper; bound it in WP6 with the two extremes (10 m-only ≈ lower bound, full transfer = upper bound).

### Steps

**A. Station correction table (~1–2 d).** For each of the ~200 stations, pair hourly ERA5-10 m with observed 10 m over **2015–2022**:
- RMSE → ERA5 quality class: Class 1 ≤ 1.5 m/s, Class 2 1.5–3, Class 3 > 3 (Hu 2023 thresholds).
- 13 empirical quantiles (P5, P12.5, …, P95) of both series → per-quantile correction ratios.

**B. Classifier map (~2–3 d).** Random-forest classifier: topo features (elevation, slope, aspect, TPI@5/75 km, TDI, z₀; optionally mean `gwd`) → quality class. Train on the ~200 stations (80/20 split), then predict for all 160 sites + 13 parks + a Germany map. *Deliverable:* the applicability-domain figure (Hu replication for your sites).

**C. Branch-B correction model (~3–5 d).** Regionalized quantile mapping à la Houndekindo 2024:
- Model: LightGBM **quantile regression** (pinball loss), one model per quantile (13 total).
- Features: topo metrics + ERA5 10 m quantiles at the location (+ optionally mean gwd). Target: station 10 m quantiles.
- Validation: **leave-station-out spatial CV** (never random k-fold — spatial leakage). Report skill vs (i) raw ERA5, (ii) nearest-station QM baseline.
- Transparent fallback if LGBM under-delivers: regression-kriging of the quantile ratios (PyKrige), same CV.

**D. Gated application + park re-validation (~2–3 d).**

```text
IF   nearest station ≤ 15 km AND terrain-similar (|Δelev-std| ≤ ~20 m)
     → Branch A: nearest-station empirical QM
ELIF predicted class ≥ 2
     → Branch B: model-predicted local quantiles
ELSE → Branch C: no correction
```

- QM application per hour: `v_corr = F_loc⁻¹( F_ERA5(v) )` via the 13-point quantile map (linear interpolation between quantiles, linear tail extension).
- Apply height-consistently (rule above) → rerun the power chain → evaluate with WP1 **per branch**, including the expected nulls.
- **Falsifiable predictions (write in the lab notebook before running):** Lower Saxony improves via A; Bavaria/Palatinate/Hesse South improve via B; Schleswig-Holstein West & Mecklenburg-WP unchanged via C.

**Definition of done:** classifier figure; LSO-CV skill table; per-branch before/after park table.

---

## WP3 — Stability-corrected extrapolation (MOST)

Replaces the neutral-shear assumption above 100 m with a stability-aware profile. Uses only ERA5 single-level fields (u*, `sshf`, T, ρ already in the chain; `blh` new).

**3.1 Obukhov length per timestep:**
```
H  = −sshf/3600                        # W/m², upward positive
L  = −(u*³ · ρ · c_p · T) / (κ · g · H)   # c_p=1005, κ=0.4, g=9.81
```
Guard: `|H| < 5 W/m²` → treat as neutral (ψ_m = 0). Cap `z/L` at ±2.

**3.2 Businger–Dyer stability functions:**
- Unstable (L < 0): `x = (1 − 16·z/L)^(1/4)`; `ψ_m = ln[((1+x²)/2)·((1+x)/2)²] − 2·atan(x) + π/2`
- Stable (L > 0): `ψ_m = −5·z/L`

**3.3 Profile anchored at 100 m** (shortest extrapolation path):
```
v(h) = v100 · [ln(h/z₀) − ψ_m(h/L)] / [ln(100/z₀) − ψ_m(100/L)]
```
z₀ from CORINE (WP0.5). Apply the WP2 correction *before* this step (corrected v100 in, consistent).

**3.4 BLH applicability flag:** hours with `hub_height > 0.1·blh` → flag as outside the surface-layer validity; report the flagged share per park and metrics with/without.

**3.5 Adoption rule:** run as experiment `+stab` vs the current power-law shear; **adopt only if the 13-park power validation improves**; either way report the stability-stratified error table (bin errors by L: unstable / neutral / stable) — that table alone answers the "neutral assumption" critique.

**Definition of done:** `+stab` variant switchable in config; stability-stratified error table; BLH-flag statistics.

---

## WP4 — Aging v2: Weibull wear-out + EEG-step test + fleet recalibration

**4.1 New degradation schedule (drop-in replacement for `DF = (1−ADR)^age`):**
```python
LAMBDA, KAPPA = 54.5, 2.0        # λ anchored: 20-yr loss = 20×0.63% = 12.6% (Germer-equivalent)
def DF(age):                      # age in years (float)
    return np.exp(-(age/LAMBDA)**KAPPA)
```
Everything downstream unchanged (`v_aged = v·(1/DF)^(1/3)`). Reference numbers to unit-test against: loss = 0.8 % @ 5 y, 3.3 % @ 10 y, 12.6 % @ 20 y, 23.2 % @ 28 y; annual rate 0.34/0.67/1.35/1.89 %/yr @ 5/10/20/28 y.

**4.2 EEG-step variant (German year-20 hypothesis):**
```python
def DF_step(age, delta=0.02):
    return DF(age) * (1-delta)**max(age-20, 0)
```

**4.3 Cross-sectional fleet recalibration/validation** (uses the *final* WP2/WP3 wind):
- Per park: `efficiency = Σ P_meas / Σ P_synth(no aging)` over the validation window.
- Regress `ln(efficiency)` on park age (N = 13). Compare three curves: constant-ADR, Weibull, Weibull+step (δ free or fixed 2–4 %).
- Report: slope/CI vs Germer's 0.63 %/yr; per-park residuals; AIC/looCV between the three forms. **Validate, don't fit** — Weibull parameters stay literature-anchored; the regression only *tests* which shape tracks the age gradient.
- Confounder honesty (Murgia/Germer): curtailment screen from WP1.3 applied; note age⊗technology collinearity as a stated limitation.

**Definition of done:** `DF` variant switchable (const / weibull / weibull+step); the bias-vs-age figure with all three curves; comparison table.

---

## WP5 — Wake losses (PyWake, validation-only)

Scope: the published 160-site dataset stays **single-turbine free-stream** (correct by construction). Wakes enter only in the 13-park validation, where reality is multi-turbine.

**5.1 Setup:**
```python
# pip install py_wake  (pin the version; cite it)
from py_wake.literature.noj import Jensen_1983     # top-hat NOJ
from py_wake.site import UniformSite
from py_wake.wind_turbines import WindTurbine      # power + CT curves per model
```
- Per park: turbine positions (WP0.6), hub-height wind speed time series (final corrected+stab chain), wind direction from ERA5 100 m u/v.
- Wake decay constant **k = 0.075** (onshore convention); sensitivity over k ∈ {0.05, 0.075, 0.10} (feeds WP6).

**5.2 Run:** PyWake time-series mode (`wfm(x, y, wd=..., ws=..., time=True)`), chunked monthly. Output per hour: farm power with wakes vs Σ free-stream → **wake loss factor w(t) = P_waked/P_free ∈ (0,1]**.

**5.3 Apply & validate:** multiply the synthetic park power by w(t) → WP1 evaluation before/after. Deliverable: **wake-loss % table per park** (mean w vs layout density/terrain) — a publishable result on its own. Expected: the systematic overestimation vs measured park power shrinks.

**Definition of done:** `+wake` switchable; per-park wake-loss table; before/after metrics.

---

## WP6 — Global sensitivity analysis (Morris → Sobol, SALib)

**6.1 Scalar output:** mean over the 13 parks of |energy-ratio − 1| (from WP1). (Secondary: mean R².)

**6.2 Parameter table (bounds = the SA spec, reused by WP7):**

| Parameter | Range | Note |
|---|---|---|
| wind-level factor | ×[0.95, 1.05] per class-dependent σ from WP2-CV | dominates via v³ |
| correction strength | {off, 10 m-only, height-consistent} | bounds the Flaw-1 assumption |
| shear method | {power-law α, MOST} | WP3 switch |
| z₀ | ×[0.5, 2] (log-uniform) | CORINE uncertainty |
| aging λ | [45, 70] yr | ≙ 20-yr loss ~8–19 % |
| aging κ | [1.5, 2.5] | acceleration exponent |
| power-curve scale | ×[0.97, 1.03] | manufacturer tolerance |
| air density | {static 1.225, dynamic} | expected minor — report it |
| wake k | [0.05, 0.10] | WP5 |

**6.3 Morris screening:** `SALib.sample.morris` with r = 30 trajectories → (k+1)·r ≈ 300 chain runs (vectorized chain ⇒ minutes). Deliverable: μ*–σ ranking plot.
**6.4 Sobol on the top ~4 factors:** Saltelli sampling, N = 512–1024 base. Deliverable: S1/ST bar chart.
**Anticipated result (fine — use it):** wind-level accuracy dominates; air density is minor → justifies the effort allocation and repositions density honestly.

**Definition of done:** every factor switchable via config; Morris figure + Sobol table.

---

## WP7 — Uncertainty bands (OPTIONAL / stretch)

> **Triage verdict:** WP7 is the only work package **not directly demanded by any reviewer** (R1-6 demands sensitivity = WP6; statistical rigor = WP1). It is insurance + novelty, not obligation. **Mandatory minimum = 7.1 only** (the delta-method paragraph, hours of work) — it yields one quantitative limitations statement ("wind-speed uncertainty amplifies ×3 into power; first-order propagation gives ±X %") and a clean future-work hook. **7.2 (MC + coverage) is a stretch goal**: do it only if WP1–WP6 are done and time remains; otherwise bank it as the follow-up paper's core (fully calibrated probabilistic dataset).

**7.1 First-order sanity band (hours of work — DO THIS):** below rated, `(σ_P/P)² ≈ (3σ_v/v)² + (σ_ρ/ρ)² + (σ_DF/DF)² + (σ_curve/P)²` — the ×3 wind amplification is the story; breaks near cut-in/rated (use as cross-check only).

**7.2 Monte Carlo + coverage (stretch goal / follow-up paper):**
- Input distributions = WP6 table (wind σ per ERA5 class from WP2-CV; λ, κ; z₀; curve; wake k). N = 500–1000 samples/timestep, vectorized (13 parks × ~9.5 kh × 1000 ⇒ seconds–minutes in NumPy).
- Output: P5/P50/P95 per hour per park.
- **Coverage check:** share of measured hourly power inside [P5, P95] → target ≈ 90 %. If off, scale the wind σ (dominant) and **report the scaling factor as the calibration step**.
- Caveat to state: v1 samples inputs independently → understates total uncertainty (correlated sampling = future work / follow-up paper); annual-energy bands via daily block bootstrap, not hourly independence.

**Definition of done:** coverage figure (nominal vs empirical) + one park-week fan chart.

---

## Final experiment ladder (incremental, attribution-friendly)

| ID | Chain | Purpose |
|---|---|---|
| M0 | Renewables.ninja | external baseline |
| M1 | era5 + const aging *(current published)* | round-1 anchor |
| M2 | M1 + gated correction (WP2) | correction increment |
| M3 | M2 + MOST stability (WP3, if it wins) | extrapolation increment |
| M4 | M3 with Weibull aging (WP4; ±step variant) | aging increment |
| M5 | M4 + wakes (WP5, parks only) | wake increment |
| — | dwd+era5(±aging) unchanged | measurement pathway, continuity |

Each rung evaluated with WP1 (all metrics, exact Wilcoxon + effect sizes, N = 13). The M-ladder **is** the component-attribution answer, complemented by WP6.

## Paper deliverables ↔ reviewer concerns

| Deliverable | Answers |
|---|---|
| ERA5 class map (Germany + sites/parks) | R1-2, R1-5 (applicability domain) |
| LSO-CV skill table (correction model) | R1-1 ("any site" evidence, DE-bounded) |
| Per-branch before/after park validation | R1-1, R1-2 |
| Stability-stratified errors + BLH flags | R1-5 |
| Aging: bias-vs-age with 3 curves + EEG test | R1-3, R3 |
| Morris/Sobol figures | R1-6 |
| MC coverage + fan chart | R1-3, R3 (beyond "deterministic") |
| Exact Wilcoxon + effect sizes (N=13 stated) | R1-4 |
| ACF figure, energy ratio, Wasserstein | R2-1 |
| Wake-loss table per park | R2-3, round-1 wake critique |
| Curtailment/icing sensitivity + limitations | R2-3, R1-7 |

**requirements.txt (core):** `numpy pandas xarray cdsapi scipy scikit-learn lightgbm SALib py_wake statsmodels rasterio rioxarray richdem matplotlib entsoe-py`

**Rough effort:** WP0 3–4 d · WP1 2–3 d · WP2 8–12 d · WP3 3–4 d · WP4 3–4 d · WP5 4–6 d · WP6 3–4 d · WP7 3–5 d → **~6–8 weeks serial**, parallelizable to ~4–5.
