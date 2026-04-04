#!/usr/bin/env python3
"""
Download PhysBench dataset from HuggingFace.

PhysBench: 10,002 multiple-choice physics questions across 4 domains.
Source: https://huggingface.co/datasets/USC-GVL/PhysBench

The HF dataset contains questions but NOT ground truth answers.
Answers are stored separately in the PhysBench GitHub repo:
  https://github.com/USC-GVL/PhysBench/tree/main/eval/physbench

This script:
  1. Downloads questions via the `datasets` library
  2. Downloads answer files from the GitHub repo
  3. Merges answers into the question data and saves per-split JSON files

Usage:
    python scripts/download_physbench.py [--data-dir data/physbench]

Requirements:
    pip install datasets
"""

import argparse
import json
import os
import sys
import urllib.request


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


def download_questions(data_dir: str) -> list:
    """Download questions via the datasets library."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets library not installed.")
        print("Install with: pip install datasets")
        sys.exit(1)

    print("Loading questions from HuggingFace (USC-GVL/PhysBench)...")
    ds = load_dataset("USC-GVL/PhysBench", split="test")
    data = [dict(row) for row in ds]
    print(f"  Loaded {len(data)} questions")
    return data


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


def main(data_dir: str):
    os.makedirs(data_dir, exist_ok=True)

    print("=" * 60)
    print("Downloading PhysBench dataset")
    print(f"Target directory: {data_dir}")
    print("=" * 60)

    # Step 1: Download answer files from GitHub
    print("\n--- Step 1: Download answer files ---")
    answers_by_idx = download_answer_files(data_dir)

    # Step 2: Download questions from HuggingFace
    print("\n--- Step 2: Download questions ---")
    data = download_questions(data_dir)

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
    args = parser.parse_args()
    data_dir = os.path.abspath(args.data_dir)
    main(data_dir)
