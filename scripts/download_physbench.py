#!/usr/bin/env python3
"""
Download PhysBench dataset from HuggingFace.

PhysBench: 10,002 multiple-choice physics questions across 4 domains.
Source: https://huggingface.co/datasets/USC-PSI-Lab/PhysBench

Usage:
    python scripts/download_physbench.py [--data-dir data/physbench]

Requirements:
    pip install huggingface_hub
"""

import argparse
import json
import os
import subprocess
import sys


def download_with_huggingface_cli(data_dir: str):
    """Download using huggingface-cli (preferred method)."""
    os.makedirs(data_dir, exist_ok=True)

    print("=" * 60)
    print("Downloading PhysBench dataset from HuggingFace...")
    print(f"Target directory: {data_dir}")
    print("=" * 60)

    # Download the dataset files
    cmd = [
        sys.executable, "-m", "huggingface_hub", "download",
        "USC-PSI-Lab/PhysBench",
        "--repo-type", "dataset",
        "--local-dir", data_dir,
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"huggingface_hub download failed: {result.stderr}")
        print("\nTrying alternative: datasets library...")
        return download_with_datasets(data_dir)

    print(result.stdout)

    # Check what we got
    print("\n" + "=" * 60)
    print("Download complete. Contents:")
    for root, dirs, files in os.walk(data_dir):
        level = root.replace(data_dir, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        if level < 2:  # Don't go too deep
            for f in sorted(files)[:20]:
                size = os.path.getsize(os.path.join(root, f))
                size_str = f"{size / 1e6:.1f}MB" if size > 1e6 else f"{size / 1e3:.1f}KB"
                print(f"{indent}  {f} ({size_str})")
            if len(files) > 20:
                print(f"{indent}  ... and {len(files) - 20} more files")

    # Extract images/videos if zip files exist
    import zipfile
    for zname in ["image.zip", "video.zip"]:
        zpath = os.path.join(data_dir, zname)
        if os.path.exists(zpath):
            extract_dir = os.path.join(data_dir, zname.replace(".zip", ""))
            if not os.path.exists(extract_dir):
                print(f"\nExtracting {zname}...")
                with zipfile.ZipFile(zpath, "r") as zf:
                    zf.extractall(extract_dir)
                print(f"  Extracted to {extract_dir}/")

    # Validate test.json exists
    test_json = os.path.join(data_dir, "test.json")
    if os.path.exists(test_json):
        with open(test_json, "r") as f:
            data = json.load(f)
        print(f"\ntest.json loaded: {len(data)} questions")
        if data:
            print(f"Sample keys: {list(data[0].keys())}")
    else:
        print("\nWARNING: test.json not found. Check the download.")
        # Try to find it
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.endswith(".json"):
                    print(f"  Found: {os.path.join(root, f)}")

    return data_dir


def download_with_datasets(data_dir: str):
    """Fallback: download using the datasets library."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: Neither huggingface_hub nor datasets is installed.")
        print("Install with: pip install huggingface_hub datasets")
        sys.exit(1)

    os.makedirs(data_dir, exist_ok=True)
    print("Loading dataset via datasets library...")
    ds = load_dataset("USC-PSI-Lab/PhysBench")
    print(f"Dataset: {ds}")

    # Save as JSON for our evaluation script
    for split_name in ds:
        split = ds[split_name]
        out_path = os.path.join(data_dir, f"{split_name}.json")
        split.to_json(out_path)
        print(f"Saved {split_name} split: {len(split)} items → {out_path}")

    return data_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PhysBench dataset")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "physbench"),
        help="Directory to download PhysBench data into",
    )
    args = parser.parse_args()
    data_dir = os.path.abspath(args.data_dir)
    download_with_huggingface_cli(data_dir)
