#!/usr/bin/env python
"""Make sure the parsed MaStR register is available (parquets under settings mastr_dir) and print its date tag.
Order: parquets present -> unpacked export (settings mastr_export_dir or --export-dir) -> download of the current
BNetzA full export (~3 GB, only the solar files are unpacked; zip and XML are removed after parsing).

    synthre/bin/python scripts/pv/mastr_ensure.py                       # check, fill if missing
    synthre/bin/python scripts/pv/mastr_ensure.py --refresh             # fetch the current export although parquets exist
    synthre/bin/python scripts/pv/mastr_ensure.py --export-dir data/Marktstammdatenregister_20250625_25 --date 20250625
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import settings  # noqa: E402
from pv import mastr_source as MS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="date tag or auto (default: settings mastr_date)")
    ap.add_argument("--mastr-dir", default=None, help="parquet folder (default: settings mastr_dir)")
    ap.add_argument("--export-dir", default=None, help="unpacked export (default: settings mastr_export_dir)")
    ap.add_argument("--download-dir", default=None, help="zip and temporary XML (default: <mastr_dir>/../mastr_download)")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="fetch the current export even if parquets exist")
    ap.add_argument("--keep-zip", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    date = MS.ensure(a.date, a.mastr_dir, a.export_dir, a.download_dir, allow_download=not a.no_download,
                     refresh=a.refresh, keep_zip=a.keep_zip, workers=a.workers)
    for k, p in MS.parquet_paths(date, a.mastr_dir or settings.get("mastr_dir")).items():
        print(f"{k:6s} {p}")
    print(date)


if __name__ == "__main__":
    main()
