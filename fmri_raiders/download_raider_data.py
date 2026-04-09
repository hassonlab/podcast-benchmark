#!/usr/bin/env python3
"""
Download only the Raider (Haxby et al.) numpy bundle from Zenodo record 2598755.

File: raider.zip (~32 MB) — not the full brainiak_datasets.zip.

Unpacks so ``movie.npy``, ``image.npy``, ``label.npy`` live under ``--out-dir`` (default: repo/data/raider).

Usage::
  python fmri_raiders/download_raider_data.py
  python fmri_raiders/download_raider_data.py --out-dir /tmp/raider
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

RAIDER_ZIP_URL = "https://zenodo.org/records/2598755/files/raider.zip?download=1"

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Raider .npy data from Zenodo")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(_PROJECT_ROOT, "data", "raider"),
        help="Directory where movie.npy / image.npy / label.npy should end up",
    )
    parser.add_argument(
        "--cache-zip",
        default="",
        help="Optional path to save raider.zip (default: temp file next to out-dir)",
    )
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.cache_zip:
        zip_path = os.path.abspath(args.cache_zip)
    else:
        zip_path = os.path.join(out_dir, ".raider.zip.partial")

    print(f"Downloading {RAIDER_ZIP_URL}")
    print(f"  → {zip_path}")
    req = urllib.request.Request(
        RAIDER_ZIP_URL,
        headers={"User-Agent": "podcast-benchmark-fmri_raiders/1.0"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        chunk = 256 * 1024
        read = 0
        with open(zip_path, "wb") as f:
            while True:
                b = resp.read(chunk)
                if not b:
                    break
                f.write(b)
                read += len(b)
                if total:
                    print(f"\r  {read / total * 100:.1f}%", end="", flush=True)
    if total:
        print()

    print("Extracting …")
    required = ("movie.npy", "image.npy", "label.npy")
    with zipfile.ZipFile(zip_path, "r") as zf:
        with tempfile.TemporaryDirectory() as tmp:
            zf.extractall(tmp)
            src_dir = None
            for root, _, files in os.walk(tmp):
                if all(os.path.isfile(os.path.join(root, n)) for n in required):
                    src_dir = root
                    break
            if src_dir is None:
                print("Could not find movie.npy, image.npy, label.npy inside the zip.", file=sys.stderr)
                sys.exit(1)
            os.makedirs(out_dir, exist_ok=True)
            for n in required:
                shutil.copy2(os.path.join(src_dir, n), os.path.join(out_dir, n))

    if not args.cache_zip:
        try:
            os.remove(zip_path)
        except OSError:
            pass

    missing = [n for n in required if not os.path.isfile(os.path.join(out_dir, n))]
    if missing:
        print(f"Missing after extract: {missing}", file=sys.stderr)
        sys.exit(1)

    print(f"Ready: {out_dir}")
    print(f"  Run: python fmri_raiders/run_tutorial.py --data-dir {out_dir}")


if __name__ == "__main__":
    main()
