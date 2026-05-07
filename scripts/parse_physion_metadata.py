#!/usr/bin/env python3
"""
Parse Physion++ readout dataset metadata.

Loads .pkl files from the Physion++ readout_data.zip, extracts per-object
physics properties (mass, dynamic_friction, static_friction, bounciness),
and reports statistics. Also extracts _map.png frames as representative
images for probing.

Usage:
    python scripts/parse_physion_metadata.py
"""

import io
import json
import pickle
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ZIP_PATH = DATA_DIR / "physion_readout.zip"
OUTPUT_DIR = PROJECT_ROOT / "results" / "physion_parsed"


def parse_physion_readout():
    """Parse the Physion++ readout zip and extract physics metadata."""
    if not ZIP_PATH.exists():
        print(f"Physion++ zip not found at {ZIP_PATH}")
        return None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    zf = zipfile.ZipFile(str(ZIP_PATH), "r")
    names = zf.namelist()

    pkl_files = sorted([n for n in names if n.endswith(".pkl")])
    map_files = sorted([n for n in names if n.endswith("_map.png")])
    id_json_files = sorted([n for n in names if n.endswith("_id.json")])

    print(f"Physion++ readout dataset:")
    print(f"  PKL files (physics data): {len(pkl_files)}")
    print(f"  Map PNG files (frames): {len(map_files)}")
    print(f"  ID JSON files (segmentation): {len(id_json_files)}")

    # Parse all pkl files
    all_records = []
    scenario_stats = defaultdict(lambda: {"count": 0, "masses": [], "frictions": [], "bounces": []})

    for i, pkl_path in enumerate(pkl_files):
        parts = pkl_path.split("/")
        scenario_type = parts[1] if len(parts) > 1 else "unknown"
        trial_name = parts[2] if len(parts) > 2 else "unknown"

        try:
            with zf.open(pkl_path) as f:
                data = pickle.load(io.BytesIO(f.read()))
        except Exception as e:
            print(f"  Error loading {pkl_path}: {e}")
            continue

        static = data.get("static", {})

        masses = static.get("mass", np.array([]))
        dyn_friction = static.get("dynamic_friction", np.array([]))
        sta_friction = static.get("static_friction", np.array([]))
        bounciness = static.get("bounciness", np.array([]))
        colors = static.get("color", np.array([]))
        model_names = static.get("model_names", np.array([]))
        object_ids = static.get("object_ids", np.array([]))
        n_objects = len(masses) if hasattr(masses, '__len__') else 0

        # Find corresponding map file (first frame)
        trial_prefix = pkl_path.rsplit("/", 1)[0] if "/" in pkl_path else ""
        frame_num = Path(pkl_path).stem  # e.g., "0038"
        map_name = f"{trial_prefix}/{frame_num}_map.png"

        record = {
            "pkl_path": pkl_path,
            "scenario": scenario_type,
            "trial": trial_name,
            "frame_num": frame_num,
            "map_png": map_name if map_name in names else None,
            "n_objects": n_objects,
            "mass": masses.tolist() if hasattr(masses, 'tolist') else [],
            "dynamic_friction": dyn_friction.tolist() if hasattr(dyn_friction, 'tolist') else [],
            "static_friction": sta_friction.tolist() if hasattr(sta_friction, 'tolist') else [],
            "bounciness": bounciness.tolist() if hasattr(bounciness, 'tolist') else [],
        }
        all_records.append(record)

        # Accumulate stats
        scenario_stats[scenario_type]["count"] += 1
        if hasattr(masses, 'tolist'):
            scenario_stats[scenario_type]["masses"].extend(masses.tolist())
        if hasattr(dyn_friction, 'tolist'):
            scenario_stats[scenario_type]["frictions"].extend(dyn_friction.tolist())
        if hasattr(bounciness, 'tolist'):
            scenario_stats[scenario_type]["bounces"].extend(bounciness.tolist())

        if (i + 1) % 100 == 0:
            print(f"  Parsed {i+1}/{len(pkl_files)} PKL files", flush=True)

    print(f"\n  Total records: {len(all_records)}")

    # Print scenario statistics
    print("\n" + "=" * 70)
    print("  PHYSION++ SCENARIO STATISTICS")
    print("=" * 70)

    for scenario, stats in sorted(scenario_stats.items()):
        masses = np.array(stats["masses"])
        frictions = np.array(stats["frictions"])
        bounces = np.array(stats["bounces"])

        print(f"\n  {scenario} ({stats['count']} trials):")
        if len(masses) > 0:
            # Filter out extreme values (some objects are kinematic with mass=1000)
            real_masses = masses[(masses > 0.001) & (masses < 100)]
            print(f"    Mass: min={masses.min():.4f}, max={masses.max():.4f}, "
                  f"mean={masses.mean():.4f}")
            if len(real_masses) > 0:
                print(f"    Mass (realistic <100): min={real_masses.min():.4f}, "
                      f"max={real_masses.max():.4f}, mean={real_masses.mean():.4f}")
        if len(frictions) > 0:
            print(f"    Friction: min={frictions.min():.4f}, max={frictions.max():.4f}, "
                  f"mean={frictions.mean():.4f}")
        if len(bounces) > 0:
            print(f"    Bounciness: min={bounces.min():.4f}, max={bounces.max():.4f}, "
                  f"mean={bounces.mean():.4f}")

    # Save parsed data
    # Save just the metadata (no images) as JSON
    metadata_path = OUTPUT_DIR / "physion_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump({
            "n_records": len(all_records),
            "scenario_stats": {
                k: {
                    "count": v["count"],
                    "mass_range": [float(np.min(v["masses"])), float(np.max(v["masses"]))] if v["masses"] else [],
                    "friction_range": [float(np.min(v["frictions"])), float(np.max(v["frictions"]))] if v["frictions"] else [],
                    "bounce_range": [float(np.min(v["bounces"])), float(np.max(v["bounces"]))] if v["bounces"] else [],
                }
                for k, v in scenario_stats.items()
            },
        }, f, indent=2)

    print(f"\n  Metadata saved to {metadata_path}")

    # Extract a few sample map images for inspection
    sample_dir = OUTPUT_DIR / "sample_frames"
    sample_dir.mkdir(exist_ok=True)

    # Get one map.png from each scenario
    seen_scenarios = set()
    extracted = 0
    for rec in all_records:
        if rec["map_png"] and rec["scenario"] not in seen_scenarios:
            try:
                with zf.open(rec["map_png"]) as f:
                    img = Image.open(io.BytesIO(f.read()))
                    img.save(sample_dir / f"{rec['scenario']}_{rec['frame_num']}.png")
                    seen_scenarios.add(rec["scenario"])
                    extracted += 1
            except Exception as e:
                print(f"  Error extracting {rec['map_png']}: {e}")

    print(f"  Extracted {extracted} sample frames to {sample_dir}")
    zf.close()

    return all_records


if __name__ == "__main__":
    parse_physion_readout()
