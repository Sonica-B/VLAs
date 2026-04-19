#!/usr/bin/env python3
"""
Download PhysBench dataset from HuggingFace.

PhysBench: 10,002 multiple-choice physics questions across 4 domains.
Source: https://huggingface.co/datasets/USC-GVL/PhysBench

The HF dataset contains questions but NOT ground truth answers.
Answers are stored separately in the PhysBench GitHub repo:
  https://github.com/USC-GVL/PhysBench/tree/main/eval/physbench

This script:
  1. Downloads raw JSONs + image.zip + video.zip via huggingface_hub
     (NOT via `datasets.load_dataset` — that fails with an Arrow conversion
      error because PhysBench's test.json has mixed list/scalar columns).
  2. Extracts image.zip and video.zip.
  3. Downloads answer files from the GitHub repo.
  4. Merges answers into the per-split JSON files.

Usage:
    python scripts/download_physbench.py [--data-dir data/physbench]
    python scripts/download_physbench.py --skip-media  # JSONs only (no image/video)

Requirements:
    pip install huggingface_hub
"""

import argparse
import json
import os
import shutil
import sys
import urllib.request
import zipfile


ANSWER_URLS = {
    "val": "https://raw.githubusercontent.com/USC-GVL/PhysBench/main/eval/physbench/val_answer.json",
    "test": "https://raw.githubusercontent.com/USC-GVL/PhysBench/main/eval/physbench/test_answer.json",
}


def download_answer_files(data_dir: str) -> dict:
    """Download ground truth answer files from the PhysBench GitHub repo.

    Returns a dict mapping idx -> answer metadata.
    """
    answers_by_idx = {}
    for split_name, url in ANSWER_URLS.items():
        out_path = os.path.join(data_dir, f"{split_name}_answer.json")
        if not os.path.exists(out_path):
            print(f"Downloading {split_name} answers from GitHub...")
            urllib.request.urlretrieve(url, out_path)
        with open(out_path, "r", encoding="utf-8") as f:
            answer_data = json.load(f)
        for entry in answer_data:
            answers_by_idx[entry["idx"]] = entry
        print(f"  {split_name}_answer.json: {len(answer_data)} entries")
    return answers_by_idx


def download_questions(data_dir: str, skip_media: bool = False) -> list:
    """Download raw PhysBench files via huggingface_hub (bypasses broken Arrow conversion).

    Pulls only what we need:
      - val.json, test.json, all.json
      - image.zip, video.zip (unless --skip-media)

    Returns the merged list of all questions (from all.json if present, else
    test.json + val.json concatenated).
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed.")
        print("Install with: pip install huggingface_hub")
        sys.exit(1)

    print("Pulling raw PhysBench files from HF Hub (USC-GVL/PhysBench)...")
    patterns = ["*.json"]
    if not skip_media:
        patterns += ["image.zip", "video.zip"]

    snap = snapshot_download(
        repo_id="USC-GVL/PhysBench",
        repo_type="dataset",
        allow_patterns=patterns,
    )
    print(f"  Snapshot: {snap}")

    # Copy JSONs into data_dir.
    for name in ["val.json", "test.json", "all.json"]:
        src = os.path.join(snap, name)
        dst = os.path.join(data_dir, name)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
            print(f"  copied {name}")

    # Extract image.zip + video.zip if present.
    if not skip_media:
        for zipname in ["image.zip", "video.zip"]:
            zpath = os.path.join(snap, zipname)
            target_dir_name = zipname.replace(".zip", "")
            extracted_marker = os.path.join(data_dir, target_dir_name)
            if os.path.exists(zpath) and not os.path.exists(extracted_marker):
                print(f"  extracting {zipname}...")
                with zipfile.ZipFile(zpath) as z:
                    z.extractall(data_dir)

    # Load whichever top-level questions file exists.
    for name in ["all.json", "test.json", "val.json"]:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"  loaded {name}: {len(data)} questions")
            return data
    print("ERROR: no questions JSON found after HF snapshot.")
    sys.exit(1)


def merge_and_save(data: list, answers_by_idx: dict, data_dir: str):
    """Merge answer metadata into question data, save per-split JSON arrays."""
    # Merge
    for item in data:
        idx = item["idx"]
        if idx in answers_by_idx:
            ans = answers_by_idx[idx]
            item["answer"] = ans.get("answer", "")
            item["task_type"] = ans.get("task_type", "unknown")
            item["sub_type"] = ans.get("sub_type", "unknown")
            item["ability_type"] = ans.get("ability_type", "unknown")

    # Split by the 'split' field
    splits = {}
    for item in data:
        s = item.get("split", "test")
        splits.setdefault(s, []).append(item)

    for split_name, split_data in splits.items():
        out_path = os.path.join(data_dir, f"{split_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False)
        has_answers = sum(1 for d in split_data if d.get("answer"))
        print(f"  Saved {split_name}.json: {len(split_data)} questions, {has_answers} with answers")

    # Also save the full merged dataset
    all_path = os.path.join(data_dir, "all.json")
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"  Saved all.json: {len(data)} total questions")


def main(data_dir: str, skip_media: bool = False):
    os.makedirs(data_dir, exist_ok=True)

    print("=" * 60)
    print("Downloading PhysBench dataset")
    print(f"Target directory: {data_dir}")
    print("=" * 60)

    # Step 1: Download answer files from GitHub
    print("\n--- Step 1: Download answer files ---")
    answers_by_idx = download_answer_files(data_dir)

    # Step 2: Download questions from HuggingFace (raw files, not via datasets lib)
    print("\n--- Step 2: Download questions ---")
    data = download_questions(data_dir, skip_media=skip_media)

    # Step 3: Merge and save
    print("\n--- Step 3: Merge answers into questions and save ---")
    merge_and_save(data, answers_by_idx, data_dir)

    # Validation
    print("\n--- Validation ---")
    val_path = os.path.join(data_dir, "val.json")
    if os.path.exists(val_path):
        with open(val_path, "r", encoding="utf-8") as f:
            val_data = json.load(f)
        print(f"val.json: {len(val_data)} questions")
        if val_data:
            sample = val_data[0]
            print(f"  Sample keys: {list(sample.keys())}")
            print(f"  Sample answer: {sample.get('answer', 'MISSING')}")
            print(f"  Sample task_type: {sample.get('task_type', 'MISSING')}")
    else:
        print("WARNING: val.json not found!")

    print("\nDone!")
    return data_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PhysBench dataset")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "physbench"),
        help="Directory to download PhysBench data into",
    )
    parser.add_argument(
        "--skip-media",
        action="store_true",
        help="Skip image.zip/video.zip download (JSONs only — useful for a quick sanity check)",
    )
    args = parser.parse_args()
    data_dir = os.path.abspath(args.data_dir)
    main(data_dir, skip_media=args.skip_media)
