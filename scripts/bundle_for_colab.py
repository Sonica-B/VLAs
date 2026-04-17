#!/usr/bin/env python3
"""
Build a minimal Drive/HF-upload bundle for Colab A100 runs.

The full local repo is 15GB (mostly PhysBench zips + extracted images/videos).
For Week A random-control on Qwen3-VL-8B, Colab only needs:

  - cache/week1/features/qwen3-vl-8b_train/  (~34 MB)      [private, hard-won]
  - cache/week1/features/qwen3-vl-8b_val/    (~14 MB, opt) [private]

PhysBench data (15 GB) is PUBLIC. Do not upload it. The Colab notebook
re-downloads it directly from HuggingFace Hub in ~3 min.

Usage:
    # Default: Qwen3-VL only, train + val caches
    python scripts/bundle_for_colab.py

    # Include more models for Week B
    python scripts/bundle_for_colab.py --models qwen3-vl-8b qwen2.5-vl-7b internvl3-8b gemma4-e4b

    # Include val caches too (for sanity checks)
    python scripts/bundle_for_colab.py --include-val

    # Custom output path
    python scripts/bundle_for_colab.py --output /tmp/upload_bundle.zip
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
import zipfile


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["qwen3-vl-8b"],
                    help="Models whose train-split cache to include")
    ap.add_argument("--include-val", action="store_true",
                    help="Also include val caches (per-model, ~10-15 MB each)")
    ap.add_argument("--output", type=pathlib.Path,
                    default=pathlib.Path("./upload_phys_lens_bundle.zip"))
    ap.add_argument("--cache-root", type=pathlib.Path,
                    default=pathlib.Path("cache/week1/features"))
    args = ap.parse_args()

    if not args.cache_root.exists():
        print(f"ERROR: cache root not found: {args.cache_root}", file=sys.stderr)
        return 1

    t0 = time.time()
    wrote = 0
    bytes_written = 0
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=6) as z:
        for model in args.models:
            splits = ["train"]
            if args.include_val:
                splits.append("val")
            for split in splits:
                src_dir = args.cache_root / f"{model}_{split}"
                if not src_dir.exists():
                    print(f"  skip {src_dir}: not found")
                    continue
                for path in src_dir.rglob("*"):
                    if path.is_file():
                        arcname = path.relative_to(args.cache_root.parent.parent)
                        z.write(path, arcname)
                        wrote += 1
                        bytes_written += path.stat().st_size

    elapsed = time.time() - t0
    size_mb = args.output.stat().st_size / 1e6
    uncompressed_mb = bytes_written / 1e6
    print(f"\nBundle created: {args.output}")
    print(f"  files:        {wrote}")
    print(f"  uncompressed: {uncompressed_mb:.1f} MB")
    print(f"  compressed:   {size_mb:.1f} MB")
    print(f"  elapsed:      {elapsed:.1f}s")
    print(f"\nUpload this file to:")
    print(f"  Google Drive: MyDrive/PhysLens/upload_phys_lens_bundle.zip")
    print(f"  OR HF Hub:    huggingface-cli upload <user>/physlens-cache {args.output}")
    print(f"\nThe Colab notebook's 'Fetch cache' cell knows how to extract both.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
