"""Download the Massachusetts Buildings Dataset (Mnih, 2013).

The official site publishes one 1500x1500 image per link.  This script only
pulls a reproducible subset so that a CPU-only preliminary experiment is
feasible, while keeping the official train/valid/test split.

Usage:
    python scripts/download_data.py --train 60 --valid 10 --test 10 --workers 6
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://www.cs.toronto.edu/~vmnih/data/mass_buildings"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
UA = {"User-Agent": "Mozilla/5.0 (building-extraction-research)"}
SPLITS = ("train", "valid", "test")


def fetch(url: str, retries: int = 4, timeout: int = 120) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 - network retry
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed after {retries} tries: {url}: {last}")


def list_tiles(split: str) -> list[str]:
    html = fetch(f"{BASE}/{split}/sat/index.html").decode("utf-8", "ignore")
    names = re.findall(r'href="[^"]*/([^"/]+\.tiff)"', html)
    return sorted(dict.fromkeys(names))


def download_one(split: str, name: str) -> tuple[str, int]:
    sat_path = RAW / split / "sat" / name
    map_path = RAW / split / "map" / name.replace(".tiff", ".tif")
    sat_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.parent.mkdir(parents=True, exist_ok=True)
    if not sat_path.exists():
        sat_path.write_bytes(fetch(f"{BASE}/{split}/sat/{name}"))
    if not map_path.exists():
        map_path.write_bytes(fetch(f"{BASE}/{split}/map/{name.replace('.tiff', '.tif')}"))
    return name, sat_path.stat().st_size + map_path.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=60)
    parser.add_argument("--valid", type=int, default=10)
    parser.add_argument("--test", type=int, default=10)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    limits = {"train": args.train, "valid": args.valid, "test": args.test}

    for split in SPLITS:
        names = list_tiles(split)[: limits[split]]
        print(f"[{split}] {len(names)} tiles selected", flush=True)
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(download_one, split, n): n for n in names}
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    _, size = fut.result()
                    done += 1
                    print(f"  [{split} {done:03d}/{len(names)}] {name} ({size/1e6:.1f} MB)", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"  !! {name}: {exc}", file=sys.stderr, flush=True)
    print("done ->", RAW)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
