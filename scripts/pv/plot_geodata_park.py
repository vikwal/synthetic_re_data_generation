#!/usr/bin/env python
"""Static visualisation of a park's measured geometry: orthophoto, height above ground (DOM - DGM), the 1 m
module-tilt raster with per-table azimuth arrows, and the ground slope raster.

    synthre/bin/python scripts/pv/plot_geodata_park.py --park <id> --geodata-dir <dir with <id>_tables.csv/_rasters.tif> \
        --dop <tile or mosaic> --dom <...> --dgm <...> --out <png>
"""
import argparse

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--park", required=True)
    ap.add_argument("--geodata-dir", required=True)
    ap.add_argument("--dop", required=True)
    ap.add_argument("--dom", required=True)
    ap.add_argument("--dgm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    t = pd.read_csv(f"{a.geodata_dir}/{a.park}_tables.csv")
    ras = rasterio.open(f"{a.geodata_dir}/{a.park}_rasters.tif")
    slope, aspect, tilt, az = (ras.read(i) for i in (1, 2, 3, 4))
    b = ras.bounds
    ext = [b.left, b.right, b.bottom, b.top]
    with rasterio.open(a.dop) as d:
        win = from_bounds(*b, d.transform)
        img = d.read(out_shape=(3, 900, int(900 * (b.right - b.left) / (b.top - b.bottom))), window=win, boundless=True, fill_value=0)
    with rasterio.open(a.dom) as d:
        win = from_bounds(*b, d.transform)
        z = d.read(1, out_shape=(600, int(600 * (b.right - b.left) / (b.top - b.bottom))), window=win, boundless=True, fill_value=np.nan).astype(float)
        z[z < -1000] = np.nan
    with rasterio.open(a.dgm) as d:
        win = from_bounds(*b, d.transform)
        g = d.read(1, out_shape=z.shape, window=win, boundless=True, fill_value=np.nan).astype(float)
        g[g < -1000] = np.nan
    h = z - g
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes[0, 0].imshow(np.moveaxis(img, 0, -1), extent=ext)
    axes[0, 0].set_title("orthophoto (DOP20)", loc="left")
    im = axes[0, 1].imshow(np.clip(h, -0.5, 4), extent=ext, cmap="viridis")
    axes[0, 1].set_title("height above ground DOM20 - DGM1 (m)", loc="left")
    plt.colorbar(im, ax=axes[0, 1], shrink=0.7)
    im = axes[1, 0].imshow(tilt, extent=ext, cmap="magma", vmin=0, vmax=max(5, np.nanpercentile(tilt, 98)))
    axes[1, 0].set_title(f"module tilt per table (deg), 1 m raster; arrows = azimuth, {len(t)} faces", loc="left")
    plt.colorbar(im, ax=axes[1, 0], shrink=0.7)
    L = 12.0
    axes[1, 0].quiver(t.x, t.y, L * np.sin(np.radians(t.azimuth)), L * np.cos(np.radians(t.azimuth)), color="cyan", scale=1, scale_units="xy", angles="xy", width=0.004)
    im = axes[1, 1].imshow(slope, extent=ext, cmap="cividis", vmin=0, vmax=max(5, np.nanpercentile(slope, 98)))
    axes[1, 1].set_title("ground slope DGM1 (deg), 1 m raster", loc="left")
    plt.colorbar(im, ax=axes[1, 1], shrink=0.7)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    w = t.area_m2 / t.area_m2.sum()
    fig.suptitle(a.title or f"{a.park}: tilt {np.average(t.tilt, weights=w):.1f} deg, azimuth (area-weighted) {np.average(t.azimuth, weights=w):.0f} deg, height {t.height_above_ground.median():.1f} m, ground slope {np.average(t.ground_slope, weights=w):.1f} deg", x=0.01, ha="left", fontsize=11)
    fig.savefig(a.out, dpi=80, bbox_inches="tight")


if __name__ == "__main__":
    main()
