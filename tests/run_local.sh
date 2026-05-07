#!/bin/bash
# Local regression runner for macOS / Linux.
# Run BEFORE every commit to scripts/ or turing/.
#
# Usage:
#   bash tests/run_local.sh                # tier=structural (no deps)
#   bash tests/run_local.sh active         # also tries transformers class imports
#   bash tests/run_local.sh full           # also tries config-only network loads

set -e
TIER="${1:-structural}"

echo "==============================================================="
echo "Running local regression suite (tier=${TIER})"
echo "==============================================================="
python tests/regression.py --tier "${TIER}"
RC=$?

if [ ${RC} -ne 0 ]; then
    echo
    echo "*** REGRESSION FAILED -- DO NOT COMMIT ***"
    exit ${RC}
else
    echo
    echo "OK -- safe to commit."
    exit 0
fi
