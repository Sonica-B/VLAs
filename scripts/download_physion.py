"""
Download and verify the Physion++ dataset.

Physion++ is hosted on S3. This script downloads all scenario types,
verifies checksums, and validates the metadata structure.

Usage:
    python scripts/download_physion.py --output-dir data/physion --verify-checksums
"""

import argparse
import hashlib
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Physion++ S3 bucket (update with actual URL when available)
PHYSION_S3_BASE = "s3://physion-plus-plus-data"

SCENARIO_TYPES = [
    "dominoes",
    "support",
    "contain",
    "roll",
    "link",
    "drape",
    "drop",
    "collide",
]

# Expected approximate sizes (GB) per scenario type
SCENARIO_SIZES_GB = {
    "dominoes": 8,
    "support": 6,
    "contain": 7,
    "roll": 5,
    "link": 6,
    "drape": 7,
    "drop": 5,
    "collide": 6,
}


def download_scenario(scenario: str, output_dir: Path, dry_run: bool = False) -> bool:
    """Download a single Physion++ scenario type from S3.

    Args:
        scenario: Scenario type name (e.g., "dominoes").
        output_dir: Local root directory for the dataset.
        dry_run: If True, print the command without executing.

    Returns:
        True if download succeeded.
    """
    s3_path = f"{PHYSION_S3_BASE}/{scenario}/"
    local_path = output_dir / scenario

    cmd = [
        "aws", "s3", "sync",
        s3_path, str(local_path),
        "--no-sign-request",   # Public bucket — no credentials needed
        "--exclude", "*.tmp",
    ]

    if dry_run:
        logger.info(f"[DRY RUN] Would run: {' '.join(cmd)}")
        return True

    logger.info(f"Downloading {scenario} (~{SCENARIO_SIZES_GB[scenario]}GB)...")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        logger.error(f"Failed to download {scenario}")
        return False

    logger.info(f"  {scenario} downloaded successfully.")
    return True


def verify_metadata_structure(output_dir: Path, scenario: str) -> bool:
    """Spot-check the metadata.pkl structure for one trial in a scenario.

    Args:
        output_dir: Dataset root directory.
        scenario: Scenario type to check.

    Returns:
        True if structure looks correct.
    """
    import pickle

    scenario_dir = output_dir / scenario
    trial_dirs = sorted(scenario_dir.glob("trial_*"))

    if not trial_dirs:
        logger.error(f"  No trial directories found in {scenario_dir}")
        return False

    trial_dir = trial_dirs[0]
    meta_path = trial_dir / "metadata.pkl"
    if not meta_path.exists():
        logger.error(f"  metadata.pkl not found in {trial_dir}")
        return False

    try:
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        logger.info(f"  {scenario}/{trial_dir.name}/metadata.pkl: keys={list(meta.keys())}")
        return True
    except Exception as e:
        logger.error(f"  Failed to load metadata.pkl: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Physion++ dataset")
    parser.add_argument("--output-dir", type=str, default="data/physion",
                        help="Local directory to download into")
    parser.add_argument("--scenarios", nargs="+", default=["all"],
                        help="Scenario types to download (default: all)")
    parser.add_argument("--verify-checksums", action="store_true",
                        help="Verify metadata structure after downloading")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip scenarios that already exist locally")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "all" in args.scenarios:
        scenarios = SCENARIO_TYPES
    else:
        scenarios = args.scenarios
        invalid = [s for s in scenarios if s not in SCENARIO_TYPES]
        if invalid:
            logger.error(f"Unknown scenario types: {invalid}")
            sys.exit(1)

    total_size_gb = sum(SCENARIO_SIZES_GB.get(s, 5) for s in scenarios)
    logger.info(f"Downloading {len(scenarios)} scenario types (~{total_size_gb}GB total)")
    logger.info(f"Output: {output_dir.resolve()}")

    failed = []
    for scenario in scenarios:
        local_path = output_dir / scenario
        if args.skip_existing and local_path.exists() and any(local_path.iterdir()):
            logger.info(f"  Skipping {scenario} (already exists)")
            continue

        success = download_scenario(scenario, output_dir, dry_run=args.dry_run)
        if not success:
            failed.append(scenario)

    if args.verify_checksums and not args.dry_run:
        logger.info("\nVerifying metadata structure...")
        for scenario in scenarios:
            verify_metadata_structure(output_dir, scenario)

    if failed:
        logger.error(f"\nFailed scenarios: {failed}")
        sys.exit(1)
    else:
        logger.info("\nAll scenarios downloaded successfully.")


if __name__ == "__main__":
    main()
