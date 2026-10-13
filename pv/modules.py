"""Module level: cell temperature, DC power (PVWatts or CEC single diode), library lookup,
era proxies.

Libraries: the pvlib-bundled SAM CEC module file (2019 edition; it still holds the 2010-2015
products such as Renesola JC245M-24, REC245PE, Canadian Solar CS6P) and the 2026 SAM CSV
(FL_Contribution/data/mastr/CEC_Modules.csv, up to 755 W, bifacial). Names are matched after
normalising to [A-Za-z0-9].
"""

import os
import re

import numpy as np
import pandas as pd
import pvlib

CEC_2026 = os.path.expanduser("~/Work/FL_Contribution/data/mastr/CEC_Modules.csv")
CEC_PARAMS = ["alpha_sc", "a_ref", "I_L_ref", "I_o_ref", "R_sh_ref", "R_s", "Adjust"]

_cache = {}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def bundled_modules() -> pd.DataFrame:
    if "bundled" not in _cache:
        _cache["bundled"] = pvlib.pvsystem.retrieve_sam("CECMod").T
    return _cache["bundled"]


def modules_2026(path: str = CEC_2026) -> pd.DataFrame:
    if "cec2026" not in _cache:
        df = pd.read_csv(path, skiprows=[1, 2], low_memory=False).set_index("Name")
        num = ["STC", "PTC", "A_c", "N_s", "I_sc_ref", "V_oc_ref", "I_mp_ref", "V_mp_ref", "alpha_sc",
               "beta_oc", "T_NOCT", "a_ref", "I_L_ref", "I_o_ref", "R_s", "R_sh_ref", "Adjust", "gamma_pmp"]
        for c in num:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        _cache["cec2026"] = df
    return _cache["cec2026"]


def _search(pat: str, library: str) -> pd.DataFrame:
    frames = []
    if library in ("both", "bundled"):
        b = bundled_modules()
        frames.append(b[b.index.map(_norm).str.contains(pat, regex=False)].assign(library="bundled"))
    if library in ("both", "2026"):
        m = modules_2026()
        frames.append(m[m.index.map(_norm).str.contains(pat, regex=False)].assign(library="2026"))
    return pd.concat(frames) if frames else pd.DataFrame()


def find_module(pattern: str, library: str = "both") -> pd.DataFrame:
    """Rows whose normalised name contains the normalised pattern; if the full string fails,
    fall back to the model token (longest token with a digit) filtered by the manufacturer's
    first word."""
    hits = _search(_norm(pattern), library)
    if not hits.empty:
        return hits
    tokens = [t for t in re.split(r"[\s,;:]+", pattern) if any(ch.isdigit() for ch in t)]
    if not tokens:
        return hits
    model = max(tokens, key=len)
    hits = _search(_norm(model), library)
    maker = _norm(pattern.split()[0])
    by_maker = hits[hits.index.map(_norm).str.contains(maker, regex=False)] if len(maker) >= 3 else hits
    return by_maker if not by_maker.empty else hits


def module_params(name: str, library: str = "both") -> pd.Series:
    """Single-diode parameter row for the first exact-or-containing match."""
    hits = find_module(name, library)
    if hits.empty:
        raise KeyError(f"module '{name}' not found in {library} CEC library")
    exact = hits[hits.index.map(_norm) == _norm(name)]
    row = (exact if not exact.empty else hits).iloc[0].copy()
    for c in CEC_PARAMS + ["STC", "V_mp_ref", "I_mp_ref", "gamma_pmp", "T_NOCT", "A_c", "N_s"]:
        if c in row:
            row[c] = float(row[c])
    row["name"] = row.name
    return row


# ------------------------------------------------------------ era proxies

# (first year, last year) -> technology label, typical module, gamma_pmp (%/K), notes
ERA_TABLE = [
    (1990, 2008, "multi_si_early", ["Kyocera KC175GT", "Sharp ND-224UC1"], -0.47),
    (2009, 2015, "multi_si", ["Canadian Solar CS6P-245P", "Renesola JC245M-24/Bx", "REC Solar REC245PE", "Yingli YL245P-29b"], -0.43),
    (2016, 2018, "multi_perc", ["Canadian Solar CS6K-275P", "Trina TSM-275PD05", "Hanwha Q CELLS Q.PLUS BFR-G4.1 275"], -0.40),
    (2019, 2021, "mono_perc", ["Canadian Solar CS3W-400P", "LONGi LR4-72HPH-450M", "JA Solar JAM72S10-400/MR"], -0.36),
    (2022, 2030, "topcon_bifacial", ["Jinko JKM550N-72HL4-BDV", "Trina TSM-690NEG21C20", "LONGi LR5-72HBD-540M", "JA Solar JAM72D40-580/MB"], -0.30),
]


def era_of(year: int) -> str:
    for y0, y1, label, _, _ in ERA_TABLE:
        if y0 <= year <= y1:
            return label
    return "topcon_bifacial" if year > 2030 else "multi_si_early"


def era_proxy_module(year: int, wp_per_module: float = None, library: str = "both") -> pd.Series:
    """Representative library module for a commissioning year (and, if known, Wp per module)."""
    cands = []
    for y0, y1, label, names, _ in ERA_TABLE:
        if y0 <= year <= y1:
            cands = names
    for n in cands:
        try:
            return module_params(n, library)
        except KeyError:
            continue
    # fall back: any module of the right wattage class
    lib = pd.concat([bundled_modules().assign(library="bundled"), modules_2026().assign(library="2026")])
    target = wp_per_module or {"multi_si_early": 175, "multi_si": 245, "multi_perc": 275, "mono_perc": 400, "topcon_bifacial": 550}[era_of(year)]
    lib = lib.dropna(subset=["STC"])
    row = lib.iloc[(lib["STC"] - target).abs().argsort().iloc[0]].copy()
    row["name"] = row.name
    return row


# ------------------------------------------------------- cell temperature

def cell_temperature(poa: pd.Series, temp_air: pd.Series, wind_speed: pd.Series, model: str = "faiman",
                     **kw) -> pd.Series:
    """Faiman (u0 25, u1 6.84 for open-rack; wind at 10 m) or PVsyst (u_c 29, u_v 0)."""
    if model == "faiman":
        return pvlib.temperature.faiman(poa, temp_air, wind_speed, u0=kw.get("u0", 25.0), u1=kw.get("u1", 6.84))
    if model == "pvsyst":
        return pvlib.temperature.pvsyst_cell(poa, temp_air, wind_speed, u_c=kw.get("u_c", 29.0), u_v=kw.get("u_v", 0.0))
    raise ValueError(f"unknown cell temperature model {model}")


# --------------------------------------------------------------- DC power

def dc_pvwatts(poa_eff: pd.Series, t_cell: pd.Series, pdc0_w: float, gamma_pdc: float = -0.0035) -> pd.Series:
    """PVWatts DC: linear in irradiance, linear temperature coefficient (1/K)."""
    return pvlib.pvsystem.pvwatts_dc(poa_eff, t_cell, pdc0_w, gamma_pdc=gamma_pdc, temp_ref=25.0).clip(lower=0.0)


def dc_cec(poa_eff: pd.Series, t_cell: pd.Series, mod: pd.Series, pdc0_w: float,
           degradation: float | pd.Series = 1.0, r_s_growth: float = 0.0) -> pd.DataFrame:
    """CEC single-diode DC at the maximum power point, scaled to pdc0_w of modules.

    degradation multiplies the light current (LID/soiling-like current loss); r_s_growth adds to
    the series resistance as a fraction (fill-factor / low-light loss of ageing). Returns p_mp (W),
    v_mp (V per module), i_mp (A per module) and the module count used."""
    n_mod = pdc0_w / float(mod["STC"])
    il, io, rs, rsh, nnsvth = pvlib.pvsystem.calcparams_cec(
        poa_eff, t_cell, alpha_sc=mod["alpha_sc"], a_ref=mod["a_ref"], I_L_ref=mod["I_L_ref"],
        I_o_ref=mod["I_o_ref"], R_sh_ref=mod["R_sh_ref"], R_s=mod["R_s"], Adjust=mod["Adjust"])
    il = il * degradation
    rs = rs * (1.0 + r_s_growth)
    sd = pvlib.pvsystem.singlediode(il, io, rs, rsh, nnsvth)
    p = sd["p_mp"].fillna(0.0).clip(lower=0.0)
    return pd.DataFrame({"p_mp": p * n_mod, "v_mp": sd["v_mp"].fillna(0.0), "i_mp": sd["i_mp"].fillna(0.0)}, index=poa_eff.index).assign(n_modules=n_mod)


def low_light_curve(mod: pd.Series, levels=(100, 200, 400, 600, 800, 1000), t_cell: float = 25.0) -> pd.Series:
    """Relative efficiency vs irradiance at fixed cell temperature (STC efficiency = 1)."""
    g = pd.Series(list(levels), dtype=float)
    out = dc_cec(g, pd.Series(t_cell, index=g.index), mod, float(mod["STC"]))
    eff = out["p_mp"].values / g.values
    return pd.Series(eff / eff[-1], index=list(levels))


def temperature_coefficient(mod: pd.Series) -> float:
    """Effective power temperature coefficient (1/K) of the single-diode model at 1000 W/m2."""
    g = pd.Series([1000.0, 1000.0])
    out = dc_cec(g, pd.Series([25.0, 50.0]), mod, float(mod["STC"]))
    return float((out["p_mp"].iloc[1] / out["p_mp"].iloc[0] - 1.0) / 25.0)
