"""
Import the ERA5 grid-point parquet files into PostgreSQL.

Source layout (one file per grid point, columns as written by grib_to_parquet.py):
    grid_47.5000_06.0000.parquet   timestamp, latitude, longitude, u_wind_10m, ...
    grid_index.parquet             (not needed, grid_points is built from the files)

Target schema:
    grid_points(id, latitude, longitude)             one row per grid point
    era5(grid_id, ts, <11 weather columns>)          PRIMARY KEY (grid_id, ts)

Usage:
    export DATABASE_URL="postgresql://era5_user:PASSWORD@localhost:5432/era5"
    python utils/import_era5_to_postgres.py --src data/grid_points
    python utils/import_era5_to_postgres.py --src data/grid_points --verify 20

Re-running is safe: grid points that are already loaded are skipped.
Requires: pandas, pyarrow, psycopg2-binary
"""
import argparse
import glob
import io
import os
import random
import sys

import numpy as np
import pandas as pd
import psycopg2

VALUE_COLS = [
    "u_wind_10m", "v_wind_10m", "u_wind_100m", "v_wind_100m",
    "wind_gust_10m", "friction_wind", "temp_2m", "pressure",
    "dew_point_2m", "wind_speed_10m", "wind_direction_10m",
]

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS grid_points (
    id        SERIAL PRIMARY KEY,
    latitude  DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    UNIQUE (latitude, longitude)
);
CREATE TABLE IF NOT EXISTS era5 (
    grid_id INTEGER NOT NULL REFERENCES grid_points(id),
    ts      TIMESTAMP NOT NULL,
    {", ".join(f"{c} DOUBLE PRECISION" for c in VALUE_COLS)},
    PRIMARY KEY (grid_id, ts)
);
"""


def list_files(src):
    files = sorted(glob.glob(os.path.join(src, "grid_*.parquet")))
    return [f for f in files if os.path.basename(f) != "grid_index.parquet"]


def get_or_create_grid_id(cur, lat, lon):
    cur.execute(
        "INSERT INTO grid_points (latitude, longitude) VALUES (%s, %s) "
        "ON CONFLICT (latitude, longitude) DO NOTHING",
        (lat, lon),
    )
    cur.execute("SELECT id FROM grid_points WHERE latitude=%s AND longitude=%s", (lat, lon))
    return cur.fetchone()[0]


def load_file(conn, path):
    df = pd.read_parquet(path)
    lat, lon = float(df["latitude"].iloc[0]), float(df["longitude"].iloc[0])
    with conn.cursor() as cur:
        gid = get_or_create_grid_id(cur, lat, lon)
        cur.execute("SELECT COUNT(*) FROM era5 WHERE grid_id=%s", (gid,))
        existing = cur.fetchone()[0]
        if existing == len(df):
            conn.commit()
            return "skipped", len(df)
        if existing:  # partial earlier load: redo this point
            cur.execute("DELETE FROM era5 WHERE grid_id=%s", (gid,))
        out = pd.DataFrame({"grid_id": gid, "ts": pd.to_datetime(df["timestamp"])})
        for c in VALUE_COLS:
            out[c] = df[c].values
        buf = io.StringIO()
        out.to_csv(buf, index=False, header=False, na_rep="")
        buf.seek(0)
        cols = ", ".join(["grid_id", "ts"] + VALUE_COLS)
        cur.copy_expert(f"COPY era5 ({cols}) FROM STDIN WITH (FORMAT csv)", buf)
    conn.commit()
    return "loaded", len(df)


def verify(conn, files, n):
    sample = random.sample(files, min(n, len(files)))
    bad = 0
    for path in sample:
        df = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
        lat, lon = float(df["latitude"].iloc[0]), float(df["longitude"].iloc[0])
        q = (
            f"SELECT ts, {', '.join(VALUE_COLS)} FROM era5 e JOIN grid_points g ON g.id=e.grid_id "
            "WHERE g.latitude=%s AND g.longitude=%s ORDER BY ts"
        )
        db = pd.read_sql(q, conn, params=(lat, lon))
        ok = (
            len(db) == len(df)
            and (db["ts"].values == pd.to_datetime(df["timestamp"]).values).all()
            and all(np.allclose(db[c].values, df[c].values, equal_nan=True) for c in VALUE_COLS)
        )
        print(("OK   " if ok else "FAIL ") + os.path.basename(path), len(df), "rows")
        bad += 0 if ok else 1
    print(f"verified {len(sample)} files, {bad} mismatches")
    return bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder with grid_*.parquet files")
    ap.add_argument("--dsn", default=os.environ.get("DATABASE_URL"), help="PostgreSQL connection string")
    ap.add_argument("--verify", type=int, default=0, metavar="N",
                    help="only compare N random files with the database, do not import")
    args = ap.parse_args()
    if not args.dsn:
        sys.exit("Set DATABASE_URL or pass --dsn")

    files = list_files(args.src)
    if not files:
        sys.exit(f"No grid_*.parquet files in {args.src}")

    conn = psycopg2.connect(args.dsn)
    if args.verify:
        sys.exit(0 if verify(conn, files, args.verify) else 1)

    with conn.cursor() as cur:
        cur.execute(SCHEMA)
    conn.commit()

    total = 0
    for i, path in enumerate(files, 1):
        status, rows = load_file(conn, path)
        total += rows if status == "loaded" else 0
        print(f"[{i}/{len(files)}] {status:7s} {os.path.basename(path)} ({rows} rows)", flush=True)
    with conn.cursor() as cur:
        cur.execute("ANALYZE grid_points; ANALYZE era5;")
    conn.commit()
    print(f"done, {total} rows inserted")


if __name__ == "__main__":
    main()
