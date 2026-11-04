"""Build a self-contained package that someone else can run on their machine.

The receiver only needs Python + this zip: it contains the code, the packaged
WHU dataset (optionally), the evaluation subsets, and a runbook with the exact
commands.  When the jobs finish they zip ``runs/`` + ``results/`` back to us.

    python scripts/export_package.py                # code only (~2 MB)
    python scripts/export_package.py --with-raw     # + raw tiles (~700 MB)
    python scripts/export_package.py --with-data    # + preprocessed npy (~8 GB)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CODE_DIRS = ["src", "scripts", "app"]
CODE_FILES = ["README.md", "requirements.txt", "requirements-whu.txt",
              "run_vectorizer.py", "start_day_runner.bat"]


def add_tree(zf: zipfile.ZipFile, base: Path, prefix: str, skip: tuple[str, ...] = ()) -> int:
    count = 0
    for path in base.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(base)
        if any(part in skip for part in rel.parts):
            continue
        if path.suffix in {".pyc"}:
            continue
        zf.write(path, f"{prefix}/{rel.as_posix()}")
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-raw", action="store_true", help="打包 data/raw（约 700 MB，可在对方机器上重建 npy）")
    parser.add_argument("--with-data", action="store_true", help="打包 data/whu_processed（约 8 GB）")
    parser.add_argument("--with-labels", action="store_true", help="打包 data/vertex_labels（GCN 训练用）")
    parser.add_argument("--with-models", action="store_true", default=True,
                        help="打包热启动权重 runs/*/best.pt（默认包含）")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d")
    out = Path(args.out or ROOT / f"BISHE_external_{stamp}.zip")

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        files = 0
        for d in CODE_DIRS:
            src = ROOT / d
            if src.exists():
                files += add_tree(zf, src, d, skip=("__pycache__",))
        for f in CODE_FILES:
            p = ROOT / f
            if p.exists():
                zf.write(p, f)
                files += 1
        for name in ("occlusion_subset.json", "shadow_subset.json"):
            p = ROOT / "data" / name
            if p.exists():
                zf.write(p, f"data/{name}")
                files += 1

        if args.with_raw and (ROOT / "data" / "raw").exists():
            print("打包 data/raw ...", flush=True)
            files += add_tree(zf, ROOT / "data" / "raw", "data/raw")
        if args.with_data:
            for name in ("train_images.npy", "train_masks.npy", "valid_images.npy",
                         "valid_masks.npy", "test_images.npy", "test_masks.npy", "meta.json"):
                p = ROOT / "data" / "whu_processed" / name
                if p.exists():
                    print(f"打包 {name} ...", flush=True)
                    zf.write(p, f"data/whu_processed/{name}")
                    files += 1
        if args.with_models:
            for name in ("whu_mt512", "whu_bieb", "whu_cpu_unet"):
                p = ROOT / "runs" / name / "best.pt"
                if p.exists():
                    print(f"打包权重 {name}/best.pt ...", flush=True)
                    zf.write(p, f"runs/{name}/best.pt")
                    files += 1
        if args.with_labels and (ROOT / "data" / "vertex_labels").exists():
            files += add_tree(zf, ROOT / "data" / "vertex_labels", "data/vertex_labels")

        zf.writestr("RUNBOOK.md", (ROOT / "docs" / "外部机器运行指南.md").read_text(encoding="utf-8")
                    if (ROOT / "docs" / "外部机器运行指南.md").exists() else "# RUNBOOK\n")

    size_mb = out.stat().st_size / 1e6
    print(f"写入 {out}  ({files} 个文件, {size_mb:.1f} MB)")
    print(f"预计解压后：代码 ~2 MB" + ("，数据 " + ("8 GB" if args.with_data else "700 MB") if (args.with_data or args.with_raw) else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
