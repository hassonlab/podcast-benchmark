#!/usr/bin/env python3
"""
Pieman2 / condensed numpy layout (Zenodo 2598755).

The published ``Pieman2.zip`` is **~2.7 GB**. It contains BrainIAK’s **NIfTI** Pieman2 tree (tutorial 10),
**not** a pre-made ``movie.npy``. This script downloads/extracts the zip; use **``--build-movie-npy``**
(or run ``fmri_pieman/build_movie_npy.py``) to create ``movie.npy`` (voxels × TR × subjects) from the
``intact1`` runs and gray-matter mask (needs **nibabel**).

1. **``--download``** — resolve a direct URL via the Zenodo API when possible, then stream the zip
   with **retries**, **HTTP Range resume** (partial ``*.part``), and fallback URLs on **504** timeouts.
   Default save path: ``--out-dir/Pieman2.zip`` (skips if the final ``.zip`` exists unless ``--force-download``).
2. **``--local-zip PATH``** — use an already-downloaded zip (no network).
3. **``--build-movie-npy``** — after extract, build ``movie.npy`` from NIfTI (requires ``pip install nibabel``).
4. **Manual** — place a ready ``movie.npy`` under ``data/pieman/`` yourself.

URL: https://zenodo.org/records/2598755/files/Pieman2.zip?download=1

Usage::

  python fmri_pieman/download_pieman_data.py --download --build-movie-npy
  python fmri_pieman/download_pieman_data.py --local-zip ~/Downloads/Pieman2.zip --build-movie-npy
  python fmri_pieman/build_movie_npy.py --pieman2-dir data/pieman/Pieman2
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fmri_pieman.build_movie_npy import build_movie_npy, find_pieman2_root

_PIEMAN2_ZENODO = "https://zenodo.org/records/2598755/files/Pieman2.zip?download=1"
_PIEMAN2_ZENODO_ALT = "https://zenodo.org/record/2598755/files/Pieman2.zip?download=1"
_ZENODO_RECORD_API = "https://zenodo.org/api/records/2598755"

_CHUNK = 4 * 1024 * 1024
_USER_AGENT = "podcast-benchmark-fmri-pieman/1.0 (+https://github.com)"
_READ_TIMEOUT = 900
_RETRYABLE_HTTP = frozenset({408, 429, 500, 502, 503, 504})
_CONTENT_RANGE_TOTAL = re.compile(r"bytes\s+\d+-\d+/(\d+)")


def _request_json(url: str, *, timeout: float = 60) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        return json.loads(raw.decode())
    except (urllib.error.URLError, ValueError, json.JSONDecodeError):
        return None


def _zenodo_api_file_self_url(*, max_tries: int = 5) -> str | None:
    """Resolve storage URL from Zenodo API (often more reliable than /records/.../files/... HTML)."""
    backoff = 3.0
    for t in range(1, max_tries + 1):
        payload = _request_json(_ZENODO_RECORD_API, timeout=60)
        if payload:
            for entry in payload.get("files", []) or []:
                if entry.get("key") == "Pieman2.zip":
                    links = entry.get("links") or {}
                    self_url = links.get("self")
                    if isinstance(self_url, str) and self_url.startswith("http"):
                        return self_url
            return None
        if t < max_tries:
            time.sleep(backoff + random.uniform(0, 1.5))
            backoff = min(backoff * 1.8, 90.0)
    return None


def _find_movie_npy(root: str) -> str | None:
    for dirpath, _, files in os.walk(root):
        if "movie.npy" in files:
            return os.path.join(dirpath, "movie.npy")
    return None


def _find_optional_pair(root: str) -> tuple[str | None, str | None]:
    img = lab = None
    for dirpath, _, files in os.walk(root):
        if "image.npy" in files and img is None:
            img = os.path.join(dirpath, "image.npy")
        if "label.npy" in files and lab is None:
            lab = os.path.join(dirpath, "label.npy")
    return img, lab


def _parse_content_range_total(header: str | None) -> int:
    if not header:
        return 0
    m = _CONTENT_RANGE_TOTAL.search(header)
    if not m:
        return 0
    try:
        return int(m.group(1))
    except ValueError:
        return 0


def _download_zip(url: str, dest: str, *, force: bool, max_attempts: int = 25) -> None:
    dest = os.path.abspath(dest)
    if os.path.isfile(dest) and not force:
        print(f"Using existing zip (skip download): {dest}")
        return

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    if force:
        for p in (tmp, dest):
            if os.path.isfile(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    urls_to_try = [url]
    if url == _PIEMAN2_ZENODO:
        print("Resolving file URL via Zenodo API …", flush=True)
        api_url = _zenodo_api_file_self_url()
        if api_url and api_url not in urls_to_try:
            urls_to_try.insert(0, api_url)
        if _PIEMAN2_ZENODO_ALT not in urls_to_try:
            urls_to_try.append(_PIEMAN2_ZENODO_ALT)

    print(f"Downloading …\n  (up to {len(urls_to_try)} URL(s), {max_attempts} attempts)\n  → {dest}")
    backoff = 6.0
    url_idx = 0
    attempt = 0
    same_url_http_fails = 0

    while True:
        attempt += 1
        if attempt > max_attempts:
            print(
                "Download failed after retries. Partial data is kept in *.part (resume on re-run).\n"
                "Alternatives: browser download, or\n"
                "  curl -fL -C - -o Pieman2.zip '<Zenodo URL>'\n"
                "then: python fmri_pieman/download_pieman_data.py --local-zip Pieman2.zip",
                file=sys.stderr,
            )
            sys.exit(1)

        current_url = urls_to_try[url_idx]
        resume_from = os.path.getsize(tmp) if os.path.isfile(tmp) else 0
        headers: dict[str, str] = {"User-Agent": _USER_AGENT}
        if resume_from > 0:
            headers["Range"] = f"bytes={resume_from}-"

        req = urllib.request.Request(current_url, headers=headers)
        if resume_from == 0:
            print(f"  URL: {current_url}", flush=True)

        try:
            resp = urllib.request.urlopen(req, timeout=_READ_TIMEOUT)
        except urllib.error.HTTPError as e:
            if e.code == 416 and resume_from > 0:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                print("  HTTP 416 — cleared partial file, retrying …", flush=True)
                backoff = 6.0
                same_url_http_fails = 0
                continue
            if e.code in _RETRYABLE_HTTP:
                same_url_http_fails += 1
                print(
                    f"  HTTP {e.code}, retry in {backoff:.0f}s (attempt {attempt}/{max_attempts}) …",
                    flush=True,
                )
                time.sleep(backoff + random.uniform(0, 2.5))
                backoff = min(backoff * 1.55, 120.0)
                if same_url_http_fails >= 4 and url_idx + 1 < len(urls_to_try):
                    url_idx += 1
                    same_url_http_fails = 0
                    backoff = 6.0
                    print(f"  Trying next URL (#{url_idx + 1} / {len(urls_to_try)}) …", flush=True)
                continue
            print(f"Download failed: {e}", file=sys.stderr)
            sys.exit(1)
        except urllib.error.URLError as e:
            print(
                f"  Network error ({e!r}), retry in {backoff:.0f}s (attempt {attempt}/{max_attempts}) …",
                flush=True,
            )
            time.sleep(backoff + random.uniform(0, 2.5))
            backoff = min(backoff * 1.55, 120.0)
            continue

        code = resp.getcode()
        same_url_http_fails = 0
        total_n = 0
        try:
            if resume_from > 0 and code == 200:
                resp.close()
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                print("  Server ignored Range — restarting from byte 0 …", flush=True)
                attempt -= 1
                continue

            cl = resp.headers.get("Content-Length")
            if cl and str(cl).isdigit():
                total_n = int(cl)
            if code == 206:
                tr = _parse_content_range_total(resp.headers.get("Content-Range", ""))
                if tr > 0:
                    total_n = tr

            append_mode = resume_from > 0 and code == 206
            open_mode = "ab" if append_mode else "wb"
            session_read = 0
            last_pct = -1
            last_mb_line = -1

            with open(tmp, open_mode) as f:
                while True:
                    chunk = resp.read(_CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    session_read += len(chunk)
                    got = resume_from + session_read
                    if total_n > 0:
                        pct = min(100, int(100 * got / total_n))
                        if pct >= last_pct + 5 or got >= total_n:
                            print(
                                f"  {pct:3d}%  ({got / (1024 * 1024):.0f} / {total_n / (1024 * 1024):.0f} MiB)",
                                flush=True,
                            )
                            last_pct = pct
                    else:
                        mb = got // (50 * 1024 * 1024)
                        if mb > last_mb_line:
                            print(f"  {got / (1024 * 1024):.0f} MiB …", flush=True)
                            last_mb_line = mb
        except Exception as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            try:
                resp.close()
            except Exception:
                pass
            print(
                f"  Transfer interrupted ({e!r}), partial saved — retry in {backoff:.0f}s …",
                flush=True,
            )
            time.sleep(backoff + random.uniform(0, 2.5))
            backoff = min(backoff * 1.35, 90.0)
            continue
        else:
            resp.close()

        final_size = os.path.getsize(tmp) if os.path.isfile(tmp) else 0
        if total_n > 0 and final_size != total_n:
            print(
                f"  Incomplete file ({final_size} / {total_n} bytes), retrying …",
                flush=True,
            )
            time.sleep(min(backoff, 30.0))
            continue

        try:
            os.replace(tmp, dest)
        except OSError as e:
            print(f"Could not finalize {dest}: {e}", file=sys.stderr)
            sys.exit(1)

        print("Download complete.")
        return


def _extract_and_copy(zip_path: str, out_dir: str, *, run_build_movie: bool) -> bool:
    """Return True if ``out_dir/movie.npy`` exists when this function returns."""
    print(f"Extracting {zip_path} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        with tempfile.TemporaryDirectory() as tmp:
            zf.extractall(tmp)
            movie = _find_movie_npy(tmp)
            if movie is not None:
                shutil.copy2(movie, os.path.join(out_dir, "movie.npy"))
                img, lab = _find_optional_pair(tmp)
                if img and lab:
                    shutil.copy2(img, os.path.join(out_dir, "image.npy"))
                    shutil.copy2(lab, os.path.join(out_dir, "label.npy"))
                    print("  Also copied image.npy and label.npy")
                return True

            pieman2 = find_pieman2_root(tmp)
            if pieman2 is None:
                print(
                    "No movie.npy and no BrainIAK Pieman2 layout (masks/avg152T1_gray_3mm.nii.gz) "
                    "inside the zip.",
                    file=sys.stderr,
                )
                sys.exit(1)

            dest_pieman = os.path.join(out_dir, "Pieman2")
            if os.path.isdir(dest_pieman):
                print(f"  Using existing {dest_pieman}")
            else:
                shutil.copytree(pieman2, dest_pieman)
                print(f"  Copied Pieman2 NIfTI tree → {dest_pieman}")

            out_movie = os.path.join(out_dir, "movie.npy")
            if run_build_movie:
                print("Building movie.npy from intact1 NIfTI runs (requires nibabel) …")
                try:
                    build_movie_npy(dest_pieman, out_movie)
                except ImportError as e:
                    print(str(e), file=sys.stderr)
                    sys.exit(1)
                return True

            print()
            print(
                "Pieman2 NIfTI data is in:\n"
                f"  {dest_pieman}\n"
                "Create movie.npy (install nibabel first: pip install nibabel):\n"
                f"  python fmri_pieman/build_movie_npy.py --pieman2-dir {dest_pieman} --out {out_movie}\n"
                "Or re-run this script with  --build-movie-npy",
            )
            return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Pieman data directory (movie.npy layout)")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(_PROJECT_ROOT, "data", "pieman"),
        help="Where to place movie.npy (and optional image.npy / label.npy)",
    )
    dl = parser.add_mutually_exclusive_group(required=False)
    dl.add_argument(
        "--download",
        action="store_true",
        help=f"Fetch Pieman2.zip from Zenodo (~2.7 GB), then extract (see --zip-path)",
    )
    dl.add_argument(
        "--local-zip",
        default="",
        metavar="PATH",
        help="Path to Pieman2.zip you already downloaded (extract + search for movie.npy)",
    )
    parser.add_argument(
        "--url",
        default=_PIEMAN2_ZENODO,
        help="Override download URL (default: Zenodo Pieman2.zip)",
    )
    parser.add_argument(
        "--zip-path",
        default="",
        metavar="PATH",
        help="Where to save the downloaded zip (default: <out-dir>/Pieman2.zip)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="With --download: re-download even if --zip-path already exists",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=25,
        metavar="N",
        help="Max connection/transient-retry rounds for --download (default: 25)",
    )
    parser.add_argument(
        "--build-movie-npy",
        action="store_true",
        help="After extract: build movie.npy from Pieman2 NIfTI (needs nibabel; see build_movie_npy.py)",
    )
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if not args.download and not args.local_zip:
        print("Choose one:", file=sys.stderr)
        print("  python fmri_pieman/download_pieman_data.py --download", file=sys.stderr)
        print("  python fmri_pieman/download_pieman_data.py --local-zip /path/to/Pieman2.zip", file=sys.stderr)
        print(f"  Or copy movie.npy directly into: {out_dir}", file=sys.stderr)
        sys.exit(1)

    if args.download and args.local_zip:
        parser.error("Use either --download or --local-zip, not both.")

    if args.download:
        zip_dest = args.zip_path.strip()
        if not zip_dest:
            zip_dest = os.path.join(out_dir, "Pieman2.zip")
        else:
            zip_dest = zip_dest if os.path.isabs(zip_dest) else os.path.join(_PROJECT_ROOT, zip_dest)
        _download_zip(
            args.url,
            zip_dest,
            force=args.force_download,
            max_attempts=max(1, args.max_attempts),
        )
        zip_path = zip_dest
    else:
        zip_path = args.local_zip if os.path.isabs(args.local_zip) else os.path.join(_PROJECT_ROOT, args.local_zip)

    if not os.path.isfile(zip_path):
        print(f"Not a file: {zip_path}", file=sys.stderr)
        sys.exit(1)

    if _extract_and_copy(zip_path, out_dir, run_build_movie=args.build_movie_npy):
        print(f"Ready: {out_dir}")
        print("  Train: python fmri_pieman/train_temporal_vae.py --config fmri_pieman/configs/pieman_temporal_vae.yml")


if __name__ == "__main__":
    main()
