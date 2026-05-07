#!/usr/bin/env python3
"""
Validate Physics QA Training Data.

Generates the image-conditioned QA dataset from Physion++ and validates it
BEFORE running the expensive QLoRA training. Checks:

1. All referenced images exist and load correctly
2. Prints 10 sample QA pairs with image details
3. Shows distribution of question types and physics properties
4. Reports per-scenario and per-property statistics
5. Validates ground truth value ranges
6. Optionally displays sample images (if --show-images)

Usage:
    python scripts/validate_training_data.py
    python scripts/validate_training_data.py --show-images
    python scripts/validate_training_data.py --qa-dir data/physics_qa_full
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_qa_pairs(qa_path: Path) -> list:
    pairs = []
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def validate_images(pairs: list) -> dict:
    """Validate all referenced images exist and load correctly."""
    from PIL import Image

    stats = {
        "total": len(pairs),
        "images_exist": 0,
        "images_missing": 0,
        "images_load_ok": 0,
        "images_load_fail": 0,
        "missing_paths": [],
        "sizes": [],
        "modes": set(),
    }

    for pair in pairs:
        image_path = pair.get("image_path", "")
        if not image_path:
            stats["images_missing"] += 1
            continue

        if os.path.exists(image_path):
            stats["images_exist"] += 1
            try:
                img = Image.open(image_path)
                stats["images_load_ok"] += 1
                stats["sizes"].append(img.size)
                stats["modes"].add(img.mode)
                img.close()
            except Exception as e:
                stats["images_load_fail"] += 1
                if len(stats.get("load_errors", [])) < 5:
                    stats.setdefault("load_errors", []).append(
                        f"{image_path}: {e}"
                    )
        else:
            stats["images_missing"] += 1
            if len(stats["missing_paths"]) < 10:
                stats["missing_paths"].append(image_path)

    return stats


def print_samples(pairs: list, n: int = 10, show_images: bool = False):
    """Print n sample QA pairs with their details."""
    from PIL import Image

    rng = np.random.RandomState(42)
    indices = rng.choice(len(pairs), size=min(n, len(pairs)), replace=False)

    print(f"\n{'=' * 70}")
    print(f"  SAMPLE QA PAIRS ({n} of {len(pairs)})")
    print(f"{'=' * 70}")

    for idx in indices:
        pair = pairs[idx]
        print(f"\n  --- Sample {idx} ---")
        print(f"  ID:       {pair.get('id', 'N/A')}")
        print(f"  Property: {pair.get('physics_property', 'N/A')}")
        print(f"  Scenario: {pair.get('scenario', 'N/A')}")

        image_path = pair.get("image_path", "")
        if image_path and os.path.exists(image_path):
            img = Image.open(image_path)
            print(f"  Image:    {image_path}")
            print(f"            Size: {img.size}, Mode: {img.mode}")
            img.close()
        else:
            print(f"  Image:    MISSING ({image_path})")

        question = pair.get("question", "")
        answer = pair.get("answer", "")
        gt = pair.get("ground_truth_values", {})

        print(f"  Question: {question[:120]}{'...' if len(question) > 120 else ''}")
        print(f"  Answer:   {answer[:150]}{'...' if len(answer) > 150 else ''}")
        print(f"  GT:       {gt}")

        if show_images and image_path and os.path.exists(image_path):
            try:
                img = Image.open(image_path).convert("RGB")
                # Save a thumbnail to validate visually
                thumb_dir = PROJECT_ROOT / "results" / "qa_validation_thumbs"
                thumb_dir.mkdir(parents=True, exist_ok=True)
                thumb_path = thumb_dir / f"sample_{idx}.png"
                img.save(str(thumb_path))
                print(f"  Thumbnail saved: {thumb_path}")
                img.close()
            except Exception as e:
                print(f"  Image load error: {e}")


def print_distributions(pairs: list):
    """Print distributions of question types and physics properties."""
    property_counts = defaultdict(int)
    scenario_counts = defaultdict(int)
    question_type_counts = defaultdict(int)

    # Infer question type from question text
    for pair in pairs:
        prop = pair.get("physics_property", "unknown")
        scenario = pair.get("scenario", "unknown")
        property_counts[prop] += 1
        scenario_counts[scenario] += 1

        q = pair.get("question", "").lower()
        if "heavier" in q or "mass" in q or "weigh" in q:
            question_type_counts["mass_comparison"] += 1
        elif "ratio" in q:
            question_type_counts["mass_ratio"] += 1
        elif "inertia" in q or "accelerat" in q:
            question_type_counts["inertia"] += 1
        elif "friction" in q or "slide" in q or "rough" in q:
            question_type_counts["friction"] += 1
        elif "bounce" in q or "elastic" in q or "restitution" in q:
            question_type_counts["elasticity"] += 1
        elif "collid" in q or "velocity change" in q:
            question_type_counts["collision"] += 1
        elif "force" in q or "moving" in q:
            question_type_counts["combined_reasoning"] += 1
        else:
            question_type_counts["other"] += 1

    print(f"\n{'=' * 70}")
    print(f"  PHYSICS PROPERTY DISTRIBUTION")
    print(f"{'=' * 70}")
    total = sum(property_counts.values())
    for prop, count in sorted(property_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        bar = "#" * int(pct / 2)
        print(f"  {prop:15s}: {count:5d} ({pct:5.1f}%) {bar}")

    print(f"\n{'=' * 70}")
    print(f"  QUESTION TYPE DISTRIBUTION")
    print(f"{'=' * 70}")
    total = sum(question_type_counts.values())
    for qtype, count in sorted(question_type_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        bar = "#" * int(pct / 2)
        print(f"  {qtype:20s}: {count:5d} ({pct:5.1f}%) {bar}")

    print(f"\n{'=' * 70}")
    print(f"  SCENARIO DISTRIBUTION")
    print(f"{'=' * 70}")
    total = sum(scenario_counts.values())
    for scenario, count in sorted(scenario_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        bar = "#" * int(pct / 2)
        print(f"  {scenario:30s}: {count:5d} ({pct:5.1f}%) {bar}")


def validate_ground_truth_ranges(pairs: list):
    """Check that ground truth values are within reasonable ranges."""
    print(f"\n{'=' * 70}")
    print(f"  GROUND TRUTH VALUE RANGES")
    print(f"{'=' * 70}")

    mass_values = []
    friction_values = []
    bounce_values = []
    ratio_values = []

    for pair in pairs:
        gt = pair.get("ground_truth_values", {})
        for k, v in gt.items():
            if isinstance(v, (int, float)) and not np.isnan(v):
                if "mass" in k:
                    mass_values.append(v)
                elif "friction" in k:
                    friction_values.append(v)
                elif "bounce" in k:
                    bounce_values.append(v)
                elif "ratio" in k:
                    ratio_values.append(v)

    for name, values in [
        ("Mass (kg)", mass_values),
        ("Friction (mu)", friction_values),
        ("Bounciness", bounce_values),
        ("Mass ratio", ratio_values),
    ]:
        if values:
            arr = np.array(values)
            print(f"\n  {name}:")
            print(f"    Count: {len(arr)}")
            print(f"    Min:   {arr.min():.4f}")
            print(f"    Max:   {arr.max():.4f}")
            print(f"    Mean:  {arr.mean():.4f}")
            print(f"    Std:   {arr.std():.4f}")
            print(f"    Median:{np.median(arr):.4f}")

            # Flag outliers
            q1 = np.percentile(arr, 25)
            q3 = np.percentile(arr, 75)
            iqr = q3 - q1
            outliers = np.sum((arr < q1 - 1.5 * iqr) | (arr > q3 + 1.5 * iqr))
            if outliers > 0:
                print(f"    Outliers: {outliers} ({outliers / len(arr) * 100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Validate physics QA training data")
    parser.add_argument("--qa-dir", type=str, default=None,
                        help="Directory containing QA JSONL files")
    parser.add_argument("--show-images", action="store_true",
                        help="Save sample image thumbnails for visual inspection")
    parser.add_argument("--generate", action="store_true", default=True,
                        help="Generate data if it doesn't exist (default: True)")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of sample QA pairs to print")
    args = parser.parse_args()

    qa_dir = Path(args.qa_dir) if args.qa_dir else PROJECT_ROOT / "data" / "physics_qa_full"

    train_path = qa_dir / "physics_qa_train.jsonl"
    val_path = qa_dir / "physics_qa_val.jsonl"

    # Generate if needed
    if not train_path.exists() or not val_path.exists():
        if args.generate:
            print("  QA data not found. Generating...")
            from scripts.run_qlora_full import generate_image_conditioned_qa
            generate_image_conditioned_qa(qa_dir)
        else:
            print(f"  ERROR: QA data not found at {qa_dir}")
            print(f"  Run with --generate or first run:")
            print(f"    python scripts/run_qlora_full.py --generate-data-only")
            sys.exit(1)

    # Load
    print(f"\n{'=' * 70}")
    print(f"  PHYSICS QA DATA VALIDATION")
    print(f"{'=' * 70}")

    train_pairs = load_qa_pairs(train_path)
    val_pairs = load_qa_pairs(val_path)
    all_pairs = train_pairs + val_pairs

    print(f"  Train file: {train_path}")
    print(f"  Val file:   {val_path}")
    print(f"  Train pairs: {len(train_pairs)}")
    print(f"  Val pairs:   {len(val_pairs)}")
    print(f"  Total pairs: {len(all_pairs)}")
    print(f"  Split ratio: {len(train_pairs) / len(all_pairs) * 100:.0f}% / "
          f"{len(val_pairs) / len(all_pairs) * 100:.0f}%")

    # Validate images
    print(f"\n  Validating images...")
    img_stats = validate_images(all_pairs)
    print(f"\n  Image Validation:")
    print(f"    Total samples:      {img_stats['total']}")
    print(f"    Images exist:       {img_stats['images_exist']}")
    print(f"    Images missing:     {img_stats['images_missing']}")
    print(f"    Images load OK:     {img_stats['images_load_ok']}")
    print(f"    Images load fail:   {img_stats['images_load_fail']}")
    if img_stats["sizes"]:
        sizes = set(img_stats["sizes"])
        print(f"    Unique sizes:       {sizes}")
    print(f"    Image modes:        {img_stats['modes']}")

    if img_stats["missing_paths"]:
        print(f"\n  Missing image paths (first 10):")
        for p in img_stats["missing_paths"][:10]:
            print(f"    {p}")

    load_errors = img_stats.get("load_errors", [])
    if load_errors:
        print(f"\n  Image load errors (first 5):")
        for e in load_errors[:5]:
            print(f"    {e}")

    # Coverage check
    if img_stats["images_exist"] == img_stats["total"]:
        print(f"\n  [PASS] All {img_stats['total']} images verified!")
    else:
        missing_pct = img_stats["images_missing"] / img_stats["total"] * 100
        if missing_pct > 5:
            print(f"\n  [FAIL] {missing_pct:.1f}% images missing — check data paths")
        else:
            print(f"\n  [WARN] {img_stats['images_missing']} images missing ({missing_pct:.1f}%)")

    # Print samples
    print_samples(all_pairs, n=args.num_samples, show_images=args.show_images)

    # Distributions
    print_distributions(all_pairs)

    # Ground truth ranges
    validate_ground_truth_ranges(all_pairs)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  VALIDATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total QA pairs:     {len(all_pairs)}")
    print(f"  Train / Val split:  {len(train_pairs)} / {len(val_pairs)}")
    print(f"  Images verified:    {img_stats['images_load_ok']} / {img_stats['total']}")

    unique_scenarios = len(set(p.get("scenario", "") for p in all_pairs))
    unique_props = len(set(p.get("physics_property", "") for p in all_pairs))
    print(f"  Unique scenarios:   {unique_scenarios}")
    print(f"  Physics properties: {unique_props}")

    all_ok = (
        img_stats["images_missing"] == 0
        and img_stats["images_load_fail"] == 0
        and len(all_pairs) > 100
    )

    if all_ok:
        print(f"\n  STATUS: READY FOR TRAINING")
        print(f"  Run: python scripts/run_qlora_full.py --condition merger --epochs 3")
    else:
        issues = []
        if img_stats["images_missing"] > 0:
            issues.append(f"{img_stats['images_missing']} missing images")
        if img_stats["images_load_fail"] > 0:
            issues.append(f"{img_stats['images_load_fail']} image load failures")
        if len(all_pairs) < 100:
            issues.append(f"only {len(all_pairs)} QA pairs (need >= 100)")
        print(f"\n  STATUS: ISSUES FOUND: {', '.join(issues)}")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
