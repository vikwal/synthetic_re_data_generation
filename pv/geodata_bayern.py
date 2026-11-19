"""Bavarian open geodata (CC BY 4.0, Bayerische Vermessungsverwaltung) for module geometry: DOP20 orthophoto,
DOM20 surface model and DGM1 terrain model as 1 km x 1 km UTM32 tiles. Tile URL schemes learned from the
municipality metalinks of geodaten.bayern.de/opengeodata (2026-10-14):
  DOP20 RGB  https://download1.bayernwolke.de/a/dop20/data/32<E>_<N>.tif           (~50-70 MB, 0.2 m, 3 bands)
  DOM20      https://download1.bayernwolke.de/a/dom20/DOM/32<E>_<N>_20_DOM.tif      (~45 MB, 0.2 m float32)
  DGM1       https://download1.bayernwolke.de/a/dgm/dgm1/<E>_<N>.tif               (~4 MB, 1 m float32)
with <E>, <N> the lower-left tile corner in km (EPSG:25832). TIFFTAG_DATETIME holds the acquisition date."""
import os
import subprocess

from pyproj import Transformer

BASE = "https://download1.bayernwolke.de/a"
PRODUCTS = {"dop20": "{base}/dop20/data/32{e}_{n}.tif", "dom20": "{base}/dom20/DOM/32{e}_{n}_20_DOM.tif", "dgm1": "{base}/dgm/dgm1/{e}_{n}.tif"}
_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)


def tiles_for(lat: float, lon: float, margin_m: float = 300.0) -> list:
    """(E, N) km tile keys covering a square of +-margin around the point."""
    x, y = _TO_UTM.transform(lon, lat)
    es = sorted({int((x - margin_m) // 1000), int((x + margin_m) // 1000)})
    ns = sorted({int((y - margin_m) // 1000), int((y + margin_m) // 1000)})
    return [(e, n) for e in es for n in ns]


def fetch(product: str, e: int, n: int, out_dir: str, timeout: int = 900) -> str:
    """Download one tile (skips existing files). Returns the local path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{product}_{e}_{n}.tif")
    if os.path.exists(path) and os.path.getsize(path) > 10000:
        return path
    url = PRODUCTS[product].format(base=BASE, e=e, n=n)
    subprocess.run(["curl", "-s", "-f", "-m", str(timeout), "-o", path + ".part", url], check=True)
    os.replace(path + ".part", path)
    return path
