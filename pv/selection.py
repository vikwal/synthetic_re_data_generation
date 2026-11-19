"""Selection of real open-ground PV parks from the MaStR pool for the parks_pv_v1 release, mirroring the
wind selection (FL_Contribution/pipeline/mastr/select_real_parks_v1.py): five composition classes, two
parks per class per client, four regional clients (k-means regions) + four nationwide clients + holdout.

PV classes are geometry classes from the register (the PV-specific source of heterogeneity):
  C1 south, tilt 20-40 (or 40-60) deg  C2 south, tilt < 20 deg  C3 east-west  C4 off-south main orientation
  C5 several orientations (secondary orientation, mixed unit classes or non-uniform flag)
Era cohort, size bin and hilliness (elevation std within 5 km) are spread criteria inside a client x class
pair, as the cohort was for wind. Tracking systems, north-facing, facade, > 60 deg and parks whose units
are spread over more than 1 km are excluded. Lokationen of the same operator within 500 m are merged into
one park (construction phases registered separately); a foreign park within 150 m (one site split over
several companies) makes the footprint ambiguous and is excluded, foreign neighbours up to 500 m are
only flagged (foreign_within_km)."""
import numpy as np
import pandas as pd

CLASSES = ["C1_south_std", "C2_south_flat", "C3_east_west", "C4_off_south", "C5_multi"]
OFF_SOUTH = {"Süd-Ost", "Süd-West", "Ost", "West"}
EXCLUDED_ORIENTATION = {"nachgeführt", "Nord", "Nord-Ost", "Nord-West"}
EXCLUDED_TILT = {"Nachgeführt", "> 60 Grad", "Fassadenintegriert"}
SIZE_BINS = [0.0, 2.0, 5.0, 20.0, np.inf]
SIZE_LABELS = ["<2MW", "2-5MW", "5-20MW", ">=20MW"]
TERRAIN_BINS = [-np.inf, 15.0, 40.0, np.inf]           # elevation std (m) within 5 km
TERRAIN_LABELS = ["flat", "rolling", "hilly"]


def cohort(year) -> str:
    return "<=2012" if year <= 2012 else "2013-20" if year <= 2020 else "2021+"


def pv_class(r) -> str:
    multi = bool(r.has_secondary) or bool(r.mixed_orientation) or ("0" in str(r.uniform_flag or ""))
    if multi:
        return "C5_multi"
    if r.orientation_class == "Ost-West":
        return "C3_east_west"
    if r.orientation_class in OFF_SOUTH:
        return "C4_off_south"
    if r.orientation_class == "Süd":
        return "C2_south_flat" if r.tilt_class == "< 20 Grad" else "C1_south_std"
    return None


def _km_xy(lat, lon):
    la, lo = np.deg2rad(np.asarray(lat)), np.deg2rad(np.asarray(lon))
    return np.c_[6371 * np.cos(la) * lo, 6371 * la]


def _pairs(lat, lon, radius_km: float) -> np.ndarray:
    from scipy.spatial import cKDTree
    return cKDTree(_km_xy(lat, lon)).query_pairs(radius_km, output_type="ndarray")


def merge_same_operator(parks: pd.DataFrame, units: pd.DataFrame, eeg: pd.DataFrame = None, radius_km: float = 0.5) -> pd.DataFrame:
    """Re-aggregate units so that Lokationen of one operator within radius_km form a single park (park_id =
    Lokation with the largest capacity, members listed in merged_ids)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    from pv import mastr
    p = parks.reset_index(drop=True)
    pr = _pairs(p.lat, p.lon, radius_km)
    same = p.operator.values[pr[:, 0]] == p.operator.values[pr[:, 1]]
    A = sp.coo_matrix((np.ones(same.sum()), (pr[same, 0], pr[same, 1])), shape=(len(p), len(p)))
    _, lab = connected_components(A, directed=False)
    p["cluster"] = lab
    head = p.sort_values("dc_kwp", ascending=False).drop_duplicates("cluster").set_index("cluster")["park_id"]
    p["head"] = p["cluster"].map(head)
    members = p.groupby("head")["park_id"].agg(lambda s: ";".join(sorted(s)))
    u = units[units["LokationMaStRNummer"].isin(p.park_id) | units["EinheitMastrNummer"].isin(p.park_id.str.replace("U_", ""))].copy()
    key = u["LokationMaStRNummer"].fillna("U_" + u["EinheitMastrNummer"].astype(str))
    u["LokationMaStRNummer"] = key.map(p.set_index("park_id")["head"]).fillna(key)
    merged = mastr.aggregate_parks(u, eeg)
    merged["merged_ids"] = merged["park_id"].map(members)
    merged["n_lokationen"] = merged["merged_ids"].str.count(";") + 1
    return merged


def foreign_neighbour_km(parks: pd.DataFrame, radius_km: float = 0.5) -> pd.Series:
    """Distance to the nearest park of another operator within radius_km (NaN if none)."""
    p = parks.reset_index(drop=True)
    pr = _pairs(p.lat, p.lon, radius_km)
    xy = _km_xy(p.lat, p.lon)
    out = np.full(len(p), np.nan)
    for i, j in pr:
        if p.operator.values[i] != p.operator.values[j]:
            d = float(np.hypot(*(xy[i] - xy[j])))
            out[i], out[j] = np.fmin(out[i], d), np.fmin(out[j], d)
    return pd.Series(out, index=parks.index)


def open_ground(parks: pd.DataFrame, min_kwp: float) -> pd.DataFrame:
    m = parks["mounting"].str.contains("Freifl", na=False) & (parks["status"] == "In Betrieb")
    m &= (parks["dc_kwp"] >= min_kwp) & parks["lat"].notna() & parks["lon"].notna()
    return parks[m]


def build_pool(parks: pd.DataFrame, units: pd.DataFrame, terrain: pd.DataFrame, min_kwp: float = 750.0,
               exclude_near: pd.DataFrame = None, exclude_km: float = 1.0, eeg: pd.DataFrame = None) -> pd.DataFrame:
    """Candidate pool (merged parks) with class, cohort, size bin and terrain class. exclude_near: frame with
    lat/lon whose surroundings (exclude_km) are removed, e.g. the confidential validation parks."""
    base = open_ground(parks, min_kwp)
    p = merge_same_operator(base, units, eeg)
    p = p[p["mounting"].str.contains("Freifl", na=False) & (p["status"] == "In Betrieb")]
    p = p.merge(terrain, on="park_id", how="left")        # terrain of the head Lokation (5 km metrics)
    p["foreign_within_km"] = foreign_neighbour_km(p).values
    p["reject"] = None
    p.loc[p.orientation_class.isna() | p.orientation_class.isin(EXCLUDED_ORIENTATION), "reject"] = "orientation"
    p.loc[p.reject.isna() & (p.tilt_class.isna() | p.tilt_class.isin(EXCLUDED_TILT)), "reject"] = "tilt"
    p.loc[p.reject.isna() & (p.coord_spread_km > 1.0), "reject"] = "units_spread"
    p.loc[p.reject.isna() & ~p.dc_ac.between(0.9, 2.0), "reject"] = "dc_ac"
    p.loc[p.reject.isna() & p.elev_std.isna(), "reject"] = "no_terrain"
    p.loc[p.reject.isna() & (p.foreign_within_km <= 0.15), "reject"] = "foreign_site_overlap"
    if exclude_near is not None and len(exclude_near):
        from scipy.spatial import cKDTree
        d, _ = cKDTree(_km_xy(exclude_near.lat, exclude_near.lon)).query(_km_xy(p.lat, p.lon))
        p.loc[p.reject.isna() & (d <= exclude_km), "reject"] = "validation_park"
    p["cls"] = [pv_class(r) for r in p.itertuples()]
    p.loc[p.reject.isna() & p.cls.isna(), "reject"] = "no_class"
    p["cohort"] = p["year"].map(cohort)
    p["size_bin"] = pd.cut(p["dc_kwp"] / 1e3, SIZE_BINS, labels=SIZE_LABELS, right=False).astype(str)
    p["terrain_cls"] = pd.cut(p["elev_std"], TERRAIN_BINS, labels=TERRAIN_LABELS).astype(str)
    return p


def add_regions(pool: pd.DataFrame, n_regions: int = 4, seed: int = 42) -> pd.DataFrame:
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_regions, n_init=10, random_state=seed).fit(_km_xy(pool.lat, pool.lon))
    pool = pool.copy()
    pool["region"] = km.labels_
    return pool


def select(pool: pd.DataFrame, n_regions: int = 4, seed: int = 42, per_class: int = 2) -> pd.DataFrame:
    """Greedy seeded pick: for each client and class, per_class parks; nationwide clients rotate through the
    regions; the second park of a pair prefers an unused cohort, size bin and terrain class."""
    rng = np.random.default_rng(seed)
    ok = pool[pool.reject.isna()].set_index("park_id")
    used, rows = set(), []

    def pick(c, have):
        c = c[~c.index.isin(used)]
        if len(c) == 0:
            return None
        score = (c["cohort"].isin(have["cohort"]).astype(int) + c["size_bin"].isin(have["size_bin"]).astype(int)
                 + c["terrain_cls"].isin(have["terrain_cls"]).astype(int) + rng.random(len(c)) * 0.5)
        return c.assign(score=score).sort_values("score").iloc[0]

    clients = [(f"R{r}", "regional", r) for r in range(n_regions)] + [(f"N{i}", "nationwide", None) for i in range(8 - n_regions)]
    clients.append(("holdout", "nationwide", None))
    for cid, kind, reg in clients:
        rr = 0
        for cls in CLASSES:
            have = {"cohort": set(), "size_bin": set(), "terrain_cls": set()}
            for _ in range(per_class):
                if kind == "regional":
                    cand = ok[(ok.cls == cls) & (ok.region == reg)]
                else:
                    cand = ok[(ok.cls == cls) & (ok.region == rr % n_regions)]
                    rr += 1
                pk = pick(cand, have)
                if pk is None:
                    rows.append({"client": cid, "kind": kind, "cls": cls, "park_id": None})
                    continue
                used.add(pk.name)
                for k in have:
                    have[k].add(pk[k])
                rows.append({"client": cid, "kind": kind, "cls": cls, "park_id": pk.name, "region": int(pk.region),
                             "cohort": pk.cohort, "size_bin": pk.size_bin, "terrain_cls": pk.terrain_cls,
                             "dc_kwp": round(float(pk.dc_kwp), 1), "ac_kw": round(float(pk.ac_kw), 1), "inv_kw": pk.inv_kw,
                             "dc_ac": round(float(pk.dc_ac), 3), "year": int(pk.year), "year_last": int(pk.commissioned_last.year),
                             "n_units": int(pk.n_units), "orientation_class": pk.orientation_class, "tilt_class": pk.tilt_class,
                             "secondary_orientation": pk.secondary_orientation, "secondary_tilt_class": pk.secondary_tilt_class,
                             "uniform_flag": pk.uniform_flag, "power_limit": pk.power_limit, "area_ha": pk.area_ha,
                             "n_modules": pk.n_modules, "lat": round(float(pk.lat), 5), "lon": round(float(pk.lon), 5),
                             "elevation": round(float(pk.elevation), 1), "slope": round(float(pk.slope), 2),
                             "elev_std": round(float(pk.elev_std), 1), "bundesland": pk.bundesland, "landkreis": pk.landkreis,
                             "operator": pk.operator, "n_operators": int(pk.n_operators), "eeg_tender": pk.get("eeg_tender"),
                             "n_lokationen": int(pk.n_lokationen), "merged_ids": pk.merged_ids,
                             "foreign_within_km": pk.foreign_within_km, "unit_ids": pk.unit_ids})
    return pd.DataFrame(rows)
