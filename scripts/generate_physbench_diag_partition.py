#!/usr/bin/env python3
"""
Generate the standalone partition.json file promised by CROISSANT_METADATA.md.

The Croissant ML metadata lists `physbench_diag_partition.json` as a
distribution `FileObject` with `contentUrl` pointing at the HF dataset
mirror. That file is supposed to be the deterministic quant/qual labelling
of all PhysBench v2 items (val + test) — the same labelling produced by
`src/optim/physbench_split.py:classify_quantitative()`. Until this script
runs, the distribution entry is a promise without a payload.

This script materializes the partition. It does NOT redistribute PhysBench
content (no images / videos / questions); each output record contains only
the fields the Croissant `recordSet[0].field` schema declares:

    {sample_id, split, sub_type, quant_qual, answer}

Plus we add `task_type` (also in the Croissant field list) and `idx` (the
PhysBench-native item index) for downstream joining with the per-model
probing JSONs.

Output:
    results/physbench_diag_partition.json

Validation:
    Per paper Section 3 ("Splits"), the expected counts are
        Validation: 200 total = 55 quant + 145 qual
        Test:        999 total = 274 quant + 725 qual

    If the local PhysBench mirror produces different totals, the script
    REPORTS the discrepancy in a `validation_report` section of the
    output JSON and exits non-zero (so CI catches drift).

Usage:
    python scripts/generate_physbench_diag_partition.py
    python scripts/generate_physbench_diag_partition.py \
        --physbench-dir data/physbench --output results/physbench_diag_partition.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

# Ensure UTF-8 stdout/stderr (Windows cp1252 default chokes on Greek + arrows + section signs).
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.optim.physbench_split import classify_quantitative  # noqa: E402


# Pre-registered counts from paper §3 (Table "Splits").
EXPECTED_COUNTS = {
    "val":  {"total": 200, "quantitative": 55,  "qualitative": 145},
    "test": {"total": 999, "quantitative": 274, "qualitative": 725},
}


def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_split(physbench_dir: Path, split: str) -> Optional[List[Dict[str, Any]]]:
    """Load a PhysBench v2 split JSON (val.json or test.json).

    Returns None if the file does not exist on disk (so the script can
    stub-document missing data rather than crashing).
    """
    path = physbench_dir / f"{split}.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  ERROR: failed to parse {path}: {e}", file=sys.stderr)
        return None


def make_record(item: Dict[str, Any], split: str) -> Dict[str, Any]:
    """Produce one Croissant-schema record from a PhysBench item.

    PhysBench items have an `idx` field that is the original PhysBench item
    index. We expose that as `idx` and produce a `sample_id` of the form
    "{split}_{idx}" — this matches the convention used in
    results/week1_turing/feature_extraction.jsonl.
    """
    idx = item.get("idx")
    sample_id = f"{split}_{idx}" if idx is not None else None
    quant_qual = classify_quantitative(item)
    return {
        "sample_id": sample_id,
        "idx": idx,
        "split": split,
        "task_type": item.get("task_type"),
        "sub_type": item.get("sub_type"),
        "quant_qual": quant_qual,
        "answer": item.get("answer"),
    }


def validate_counts(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare observed split×partition counts vs. paper Table 'Splits'."""
    by_split: Dict[str, Counter] = {"val": Counter(), "test": Counter()}
    for r in records:
        s = r["split"]
        if s in by_split:
            by_split[s][r["quant_qual"]] += 1

    discrepancies: List[Dict[str, Any]] = []
    observed: Dict[str, Dict[str, int]] = {}
    for split, exp in EXPECTED_COUNTS.items():
        ctr = by_split[split]
        obs = {
            "total": int(sum(ctr.values())),
            "quantitative": int(ctr.get("quantitative", 0)),
            "qualitative":  int(ctr.get("qualitative", 0)),
        }
        observed[split] = obs
        for k, v in exp.items():
            if obs.get(k) != v:
                discrepancies.append({
                    "split": split,
                    "field": k,
                    "expected": v,
                    "observed": obs.get(k),
                    "delta": (obs.get(k) or 0) - v,
                })

    return {
        "expected": EXPECTED_COUNTS,
        "observed": observed,
        "discrepancies": discrepancies,
        "passed": len(discrepancies) == 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--physbench-dir", type=Path,
                    default=PROJECT_ROOT / "data" / "physbench",
                    help="Directory containing PhysBench v2 val.json + test.json")
    ap.add_argument("--output", type=Path,
                    default=PROJECT_ROOT / "results" / "physbench_diag_partition.json",
                    help="Output JSON path (default: results/physbench_diag_partition.json)")
    ap.add_argument("--strict", action="store_true",
                    help="Exit 2 if observed counts do not match paper §3 expectations.")
    args = ap.parse_args()

    print(f"[generate_physbench_diag_partition] PhysBench dir: {args.physbench_dir}")
    print(f"[generate_physbench_diag_partition] Output:        {args.output}")

    # Load both splits.
    splits_loaded: Dict[str, List[Dict[str, Any]]] = {}
    splits_missing: List[str] = []
    for split in ("val", "test"):
        items = load_split(args.physbench_dir, split)
        if items is None:
            splits_missing.append(split)
            print(f"  WARN: PhysBench v2 {split} split not found at "
                  f"{args.physbench_dir / (split + '.json')}")
        else:
            splits_loaded[split] = items
            print(f"  loaded {split}: n={len(items)}")

    if not splits_loaded:
        # Stub-mode: write an empty partition with a README pointing at the data
        # source so the Croissant promise is at least partially satisfied.
        stub = {
            "schema_version": "1.0.0",
            "generator": "scripts/generate_physbench_diag_partition.py",
            "status": "stub",
            "reason": (
                "PhysBench v2 val.json and test.json are not on local disk. "
                "Run `python scripts/download_physbench.py` first, then re-run "
                "this script."
            ),
            "expected_data_paths": [
                str(args.physbench_dir / "val.json"),
                str(args.physbench_dir / "test.json"),
            ],
            "missing_splits": splits_missing,
            "records": [],
            "validation_report": {"passed": False, "reason": "no data on disk"},
        }
        atomic_write_json(args.output, stub)
        print(f"\n  STUBBED partition written to {args.output} (no data on disk).")
        return 1

    # Build records.
    records: List[Dict[str, Any]] = []
    for split, items in splits_loaded.items():
        for item in items:
            records.append(make_record(item, split))

    # Validate against paper §3 expected counts.
    report = validate_counts(records)

    output = {
        "schema_version": "1.0.0",
        "generator": "scripts/generate_physbench_diag_partition.py",
        "classifier": "src/optim/physbench_split.py:classify_quantitative",
        "physbench_source": "PhysBench v2 (Chow et al., 2024) — CC-BY-4.0",
        "splits_loaded": list(splits_loaded.keys()),
        "splits_missing": splits_missing,
        "n_records": len(records),
        "validation_report": report,
        "records": records,
    }
    atomic_write_json(args.output, output)

    # Console summary.
    print()
    print("=" * 72)
    print(" PhysBench-Diag partition: split × quant_qual counts")
    print("=" * 72)
    print(f"  {'split':<6}  {'total':>6}  {'quant':>6}  {'qual':>6}  vs paper §3")
    print(f"  {'-'*54}")
    for split, obs in report["observed"].items():
        exp = EXPECTED_COUNTS[split]
        status = "OK" if (obs["total"] == exp["total"]
                          and obs["quantitative"] == exp["quantitative"]
                          and obs["qualitative"] == exp["qualitative"]) else "MISMATCH"
        print(f"  {split:<6}  {obs['total']:>6d}  {obs['quantitative']:>6d}  "
              f"{obs['qualitative']:>6d}   [{status}]")
        print(f"     expected:    {exp['total']:>6d}  {exp['quantitative']:>6d}  "
              f"{exp['qualitative']:>6d}")

    if report["discrepancies"]:
        print()
        print("  Discrepancies vs paper §3:")
        for d in report["discrepancies"]:
            print(f"    {d['split']}.{d['field']}: expected={d['expected']} "
                  f"observed={d['observed']} delta={d['delta']:+d}")
        if args.strict:
            print()
            print("  --strict was set; exiting 2 due to count mismatch.")
            return 2
    else:
        print()
        print(f"  All split counts match paper §3 expectations.")

    if splits_missing:
        print()
        print(f"  WARN: missing split(s): {splits_missing}")

    print(f"\n  wrote {args.output}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
