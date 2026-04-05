"""
Qualitative vs quantitative PhysBench subset classifier.

This is the single highest-leverage utility for the Option C pivot. The core
paper claim — that the merger layer bottleneck is *specifically* for
quantitative physics — depends on splitting PhysBench into:

    qualitative: binary / categorical / comparison questions (which is heavier?
                 will it fall? does it break?) — tests whether the model knows
                 physics is happening, not the exact numbers.
    quantitative: numerical answer questions (what is the mass? the velocity?
                  how many objects? what angle?) — tests whether the model
                  extracts a numerical value from the scene.

Classification is lexical + regex-based because PhysBench does not ship a
per-sample tag for this distinction. The classifier is intentionally
conservative: when in doubt, a sample is tagged "qualitative" so the
quantitative slice is a clean subset with high precision (at the cost of some
recall). Precision matters more than recall for the core claim.

Validate by hand on 50 samples before relying on the split for the paper.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Lexical cues.
# ---------------------------------------------------------------------------

# Strong quantitative signals: question asks for a number or numerical unit.
QUANT_KEYWORDS = (
    "how many", "how much", "what is the mass", "what is the weight",
    "what is the velocity", "what is the speed", "what is the acceleration",
    "what is the angle", "what is the length", "what is the distance",
    "what is the height", "what is the width", "what is the size",
    "what is the radius", "what is the diameter", "what is the volume",
    "what is the area", "what is the density", "what is the friction",
    "what is the temperature", "what is the pressure", "what is the force",
    "count the", "number of", "estimate the", "measure the",
)

QUANT_UNITS = (
    "kg", "grams", "meters", "meter", "cm", "mm", "km", "m/s", "km/h",
    "degrees", "radians", "newtons", "joules", "pascals", "liters",
    "seconds", "minutes", "hours",
)

# Strong qualitative signals: comparison / binary / causal.
QUAL_KEYWORDS = (
    "which", "is it", "will it", "can it", "does the", "will the",
    "is the", "are the", "heavier", "lighter", "faster", "slower",
    "bigger", "smaller", "larger", "taller", "shorter",
    "more likely", "less likely", "most likely", "least likely",
    "true or false", "yes or no", "will fall", "will slide",
    "will float", "will sink", "will break",
)

NUM_PATTERN = re.compile(r"\b\d+(\.\d+)?\b")
MC_OPTION_PATTERN = re.compile(r"^\s*\(?([A-Ea-e])\)?[\.\)]\s+")


# ---------------------------------------------------------------------------
# Classifier.
# ---------------------------------------------------------------------------

def classify_quantitative(sample: Dict[str, Any]) -> str:
    """Classify a PhysBench sample as 'quantitative' or 'qualitative'.

    Input contract: the sample is a dict with at least a 'question' field
    (string) and optionally 'options' (list of strings) and 'answer' (string).
    The classifier looks at question + options + answer.

    Heuristics (in order):
        1. If the question contains a strong QUANT_KEYWORD → quantitative.
        2. If ANY option contains a unit (kg, m/s, etc.) → quantitative.
        3. If the answer is purely numeric → quantitative.
        4. If options are >50% numeric values → quantitative.
        5. Otherwise → qualitative.

    Returns one of: "quantitative", "qualitative".
    """
    question = str(sample.get("question", "")).lower()
    options = sample.get("options") or []
    options = [str(o).lower() for o in options] if isinstance(options, list) else []
    answer = str(sample.get("answer", "")).lower()

    # Strip multiple-choice letter prefixes from options for cleaner matching.
    clean_options = [MC_OPTION_PATTERN.sub("", o) for o in options]

    # Rule 1: quantitative keyword in question.
    for kw in QUANT_KEYWORDS:
        if kw in question:
            return "quantitative"

    # Rule 2: any option contains a physical unit.
    for opt in clean_options:
        for unit in QUANT_UNITS:
            # word-boundary match so "km" doesn't match "skunk"
            if re.search(rf"\b{re.escape(unit)}\b", opt):
                return "quantitative"

    # Rule 3: answer is purely numeric (e.g. "42" or "3.14").
    if answer and re.fullmatch(r"[-+]?\d+(\.\d+)?", answer.strip()):
        return "quantitative"

    # Rule 4: majority of options are numeric literals.
    if clean_options:
        numeric_options = sum(
            1 for o in clean_options if NUM_PATTERN.search(o) and not any(kw in o for kw in ("a)", "b)", "c)", "d)"))
        )
        if numeric_options / len(clean_options) > 0.5:
            return "quantitative"

    # Rule 5: default to qualitative.
    return "qualitative"


def split_physbench(
    samples: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split a list of PhysBench samples into (quantitative, qualitative).

    Returns two lists in the same order as input.
    """
    quant: List[Dict[str, Any]] = []
    qual: List[Dict[str, Any]] = []
    for s in samples:
        if classify_quantitative(s) == "quantitative":
            quant.append(s)
        else:
            qual.append(s)
    return quant, qual
