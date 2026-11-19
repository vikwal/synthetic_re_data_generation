"""Sub-arrays of a park: fixed mixtures, terrain-derived (tilt, azimuth) distributions,
Dirichlet mixtures, inter-row shading.

Angle convention: tilt from horizontal (deg), azimuth clockwise from north (deg, 180 = south).
Module normal in an east-north-up frame: n = (sin t sin a, sin t cos a, cos t).
"""

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import pvlib


@dataclass
class Array:
    name: str
    tilt: float
    azimuth: float
    share: float          # capacity share, sums to 1 over the park
    gcr: float = None     # ground coverage ratio for inter-row shading (None = off)

    def to_dict(self):
        return asdict(self)


def normal(tilt, azimuth) -> np.ndarray:
    t, a = np.radians(np.asarray(tilt, float)), np.radians(np.asarray(azimuth, float))
    return np.stack([np.sin(t) * np.sin(a), np.sin(t) * np.cos(a), np.cos(t)], axis=-1)


def angles_from_normal(n: np.ndarray) -> tuple:
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    tilt = np.degrees(np.arccos(np.clip(n[..., 2], -1, 1)))
    az = np.degrees(np.arctan2(n[..., 0], n[..., 1])) % 360.0
    return tilt, az


def effective_angles(slope: np.ndarray, aspect: np.ndarray, mount_tilt: float, mount_azimuth: float) -> tuple:
    """Rows mounted with (mount_tilt, mount_azimuth) relative to the local ground, on terrain
    with the given slope (deg) and aspect (downslope direction, deg from N): rotate the module
    normal by the terrain rotation that maps 'up' onto the ground normal."""
    slope = np.radians(np.asarray(slope, float))
    aspect = np.radians(np.asarray(aspect, float))
    nm = normal(mount_tilt, mount_azimuth)                      # (3,)
    ng = np.stack([np.sin(slope) * np.sin(aspect), np.sin(slope) * np.cos(aspect), np.cos(slope)], axis=-1)
    up = np.array([0.0, 0.0, 1.0])
    k = np.cross(np.broadcast_to(up, ng.shape), ng)               # rotation axis (horizontal)
    kn = np.linalg.norm(k, axis=-1, keepdims=True)
    k = np.where(kn > 1e-12, k / np.where(kn > 1e-12, kn, 1.0), 0.0)
    c = np.cos(slope)[..., None]
    s = np.sin(slope)[..., None]
    v = np.broadcast_to(nm, ng.shape)
    kv = np.sum(k * v, axis=-1, keepdims=True)
    rot = v * c + np.cross(k, v) * s + k * kv * (1 - c)           # Rodrigues
    return angles_from_normal(rot)


def mean_angles(tilt, azimuth, weights=None) -> tuple:
    """Capacity-weighted mean orientation (via the mean normal)."""
    n = normal(tilt, azimuth)
    w = np.ones(len(n)) if weights is None else np.asarray(weights, float)
    return angles_from_normal((n * w[:, None]).sum(axis=0))


def cluster_arrays(tilt, azimuth, weights, k: int, seed: int = 0) -> list:
    """Weighted k-means on module normals -> k Arrays with capacity shares."""
    from sklearn.cluster import KMeans
    n = normal(tilt, azimuth)
    w = np.asarray(weights, float)
    w = w / w.sum()
    k = int(min(k, len(n)))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(n, sample_weight=w)
    out = []
    for j in range(k):
        m = km.labels_ == j
        if not m.any():
            continue
        t, a = angles_from_normal((n[m] * w[m, None]).sum(axis=0))
        out.append(Array(name=f"a{j}", tilt=float(t), azimuth=float(a), share=float(w[m].sum())))
    out.sort(key=lambda x: -x.share)
    for j, a in enumerate(out):
        a.name = f"a{j}"
    return out


def dirichlet_arrays(rng: np.random.Generator, k: int, tilt0: float, az0: float, sigma_tilt: float,
                     sigma_az: float, alpha: float = 2.0) -> list:
    """Statistical terrain stand-in (federated_setting.md section 3.2 P6-P10)."""
    shares = rng.dirichlet(np.full(k, alpha))
    tilts = np.clip(rng.normal(tilt0, sigma_tilt, k), 2.0, 60.0)
    azs = (rng.normal(az0, sigma_az, k)) % 360.0
    return [Array(name=f"a{j}", tilt=float(tilts[j]), azimuth=float(azs[j]), share=float(shares[j])) for j in range(k)]


def inter_row_factors(solpos: pd.DataFrame, tilt: float, azimuth: float, gcr: float) -> pd.DataFrame:
    """Direct shaded fraction (0..1) and sky-diffuse masking factor for fixed-tilt rows with
    ground coverage ratio gcr (pitch = 1/gcr for unit collector width)."""
    pitch = 1.0 / gcr
    axis_az = (azimuth + 90.0) % 360.0                        # row axis perpendicular to the facing
    sf = pvlib.shading.shaded_fraction1d(solpos["apparent_zenith"], solpos["azimuth"], axis_az,
                                         shaded_row_rotation=tilt, collector_width=1.0, pitch=pitch,
                                         axis_tilt=0.0, surface_to_axis_offset=0.0, cross_axis_slope=0.0)
    sf = pd.Series(np.asarray(sf, float), index=solpos.index).fillna(0.0).clip(0, 1)
    mask_angle = pvlib.shading.masking_angle_passias(tilt, gcr)
    diffuse_loss = pvlib.shading.sky_diffuse_passias(mask_angle)
    return pd.DataFrame({"direct_shaded_fraction": sf, "sky_diffuse_factor": 1.0 - float(diffuse_loss)}, index=solpos.index)


def gcr_from_density(kwp_per_ha: float, wp_per_m2_module: float = 180.0) -> float:
    """Ground coverage ratio from MaStR packing density: module area / ground area."""
    return float(np.clip(kwp_per_ha * 1000.0 / 10000.0 / wp_per_m2_module, 0.15, 0.8))
