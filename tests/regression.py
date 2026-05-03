#!/usr/bin/env python3
"""
Local regression test suite for the PhysLens VLM probing pipeline.

Run BEFORE every commit that touches:
  - scripts/week1_quant_qual_probe.py (MODEL_REGISTRY, loaders)
  - scripts/extract_training_features.py (mirror registry)
  - scripts/phys_lens_predict.py (MODEL_COMPRESSION, MODEL_H3_HITS)
  - turing/*.sh (SLURM scripts)

Usage (local, no GPU/transformers needed for tier=structural):
    python tests/regression.py
    python tests/regression.py --tier structural   # default --pure Python
    python tests/regression.py --tier active       # also imports transformers, tries class import
    python tests/regression.py --tier full         # also tries Config-only loads (small downloads)

Exit codes:
    0 = all tests pass
    1 = test failure
    2 = test setup error (missing files, etc.)

This catches the class of bug that bit us 2026-05-03:
- Upgrading transformers can REMOVE support for model_types in older versions
- Some models (Qwen3-VL, InternVL3-HF, Gemma3) are only registered in
  transformers >= 4.50 / 4.51, so older versions can't load their checkpoints
- Our verify script was too strict (failed on CACHED-data models)
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
TURING = ROOT / "turing"
RESULTS = ROOT / "results"

# Models whose probing JSONs are already extracted and live in results/week1/.
# We do NOT need to load these in any current env --the JSON is the data.
CACHED_MODELS = {
    "qwen3-vl-8b",
    "qwen2.5-vl-7b",
    "internvl3-8b",
    "gemma4-e4b",
}

# Models we actively probe in v2 (must load successfully in current env).
ACTIVE_MODELS = {
    "llava-onevision-7b",
    "phi3.5-vision",
    "granite-vision-3.2-2b",  # primary 2025+ entry
    "idefics3-8b",  # backup if Granite fails
}

# Models in registry but not in either set (deprecated / dropped).
DEPRECATED_MODELS = {
    "pixtral-12b",  # broken in transformers 4.46.x, replaced by Granite/Idefics3
    "molmo-7b",     # transformers 5.x API drift, dropped
}


# =============================================================================
# Test runner
# =============================================================================

class TestSuite:
    def __init__(self, tier: str):
        self.tier = tier
        self.results: List[Tuple[str, bool, str]] = []

    def add(self, name: str, passed: bool, msg: str = "") -> None:
        self.results.append((name, passed, msg))
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}" + (f" --{msg}" if msg else ""))

    def report(self) -> int:
        passed = sum(1 for _, p, _ in self.results if p)
        failed = len(self.results) - passed
        print()
        print("=" * 72)
        print(f"REGRESSION SUITE (tier={self.tier})")
        print(f"  {passed}/{len(self.results)} passed")
        if failed:
            print(f"  {failed} FAILED:")
            for name, p, msg in self.results:
                if not p:
                    print(f"    X{name}: {msg}")
        print("=" * 72)
        return 0 if failed == 0 else 1


# =============================================================================
# Tier 1: STRUCTURAL --pure Python, no transformers dependency
# =============================================================================

def _extract_value_node(source: str, var_name: str):
    """Find the RHS AST node of a top-level assignment to var_name.

    Handles both `var = ...` (Assign) and `var: T = ...` (AnnAssign).
    Returns the value node or None if not found.
    """
    tree = ast.parse(source)
    for node in tree.body:  # only top-level
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == var_name:
                    return node.value
        elif isinstance(node, ast.AnnAssign):
            if (isinstance(node.target, ast.Name)
                    and node.target.id == var_name
                    and node.value is not None):
                return node.value
    return None


def _safe_eval(node: ast.AST):
    """Evaluate an AST node to a Python value when possible.

    Handles literals, dicts, lists, tuples, sets, simple BinOp (e.g. `2 / 3`),
    UnaryOp (`-1.0`), and Name nodes (returned as their string name).
    Falls back to a sentinel "<expr>" for unsupported expressions.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return node.id  # function reference / variable name as string
    if isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand)
        if isinstance(node.op, ast.USub) and isinstance(operand, (int, float)):
            return -operand
        return "<expr>"
    if isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
        return "<expr>"
    if isinstance(node, ast.Dict):
        return {_safe_eval(k): _safe_eval(v) for k, v in zip(node.keys, node.values)
                if k is not None}
    if isinstance(node, ast.List):
        return [_safe_eval(e) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_safe_eval(e) for e in node.elts)
    if isinstance(node, ast.Set):
        return {_safe_eval(e) for e in node.elts}
    return "<expr>"


def parse_dict_assign(source: str, var_name: str) -> Any:
    """Extract and evaluate a dict assignment (Assign or AnnAssign).

    Falls back from ast.literal_eval to _safe_eval for non-literal values
    (e.g. `2 / 3` for H3 hit-rate fractions).
    """
    val = _extract_value_node(source, var_name)
    if val is None:
        raise KeyError(f"Variable {var_name!r} not found at module top level")
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return _safe_eval(val)


def parse_dict_with_funcs(source: str, var_name: str) -> Dict[str, Any]:
    """Parse a dict where values may be function references (Name nodes)."""
    val = _extract_value_node(source, var_name)
    if val is None:
        raise KeyError(f"Variable {var_name!r} not found at module top level")
    if not isinstance(val, ast.Dict):
        raise TypeError(f"{var_name} is not a dict literal")
    out: Dict[str, Any] = {}
    for k, v in zip(val.keys, val.values):
        if k is None:
            continue
        key = _safe_eval(k)
        out[key] = _safe_eval(v)
    return out


def collect_function_names(source: str) -> set:
    """Return the set of all top-level function names in source."""
    tree = ast.parse(source)
    return {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def test_structural(suite: TestSuite) -> None:
    """Static tests that need only Python stdlib."""
    print(">> Tier: STRUCTURAL (pure Python)")

    # ---- Load source files ----
    probe_path = SCRIPTS / "week1_quant_qual_probe.py"
    extract_path = SCRIPTS / "extract_training_features.py"
    predict_path = SCRIPTS / "phys_lens_predict.py"

    for p in (probe_path, extract_path, predict_path):
        if not p.exists():
            suite.add(f"file exists: {p.name}", False, "MISSING")
            return
        suite.add(f"file exists: {p.name}", True)

    probe_src = probe_path.read_text(encoding="utf-8", errors="replace")
    extract_src = extract_path.read_text(encoding="utf-8", errors="replace")
    predict_src = predict_path.read_text(encoding="utf-8", errors="replace")

    # ---- Parse MODEL_REGISTRY (week1_quant_qual_probe.py) ----
    try:
        registry = parse_dict_assign(probe_src, "MODEL_REGISTRY")
        suite.add("parse MODEL_REGISTRY", True, f"{len(registry)} models")
    except Exception as e:
        suite.add("parse MODEL_REGISTRY", False, f"{type(e).__name__}: {e}")
        return

    # ---- Parse _LOADERS dict ----
    try:
        loaders = parse_dict_with_funcs(probe_src, "_LOADERS")
        suite.add("parse _LOADERS", True, f"{len(loaders)} loaders")
    except Exception as e:
        suite.add("parse _LOADERS", False, str(e))
        return

    # ---- Parse MODEL_COMPRESSION + MODEL_H3_HITS (phys_lens_predict.py) ----
    try:
        compression = parse_dict_assign(predict_src, "MODEL_COMPRESSION")
        suite.add("parse MODEL_COMPRESSION", True, f"{len(compression)} entries")
    except Exception as e:
        suite.add("parse MODEL_COMPRESSION", False, str(e))
        return

    try:
        h3 = parse_dict_assign(predict_src, "MODEL_H3_HITS")
        suite.add("parse MODEL_H3_HITS", True, f"{len(h3)} entries")
    except Exception as e:
        suite.add("parse MODEL_H3_HITS", False, str(e))
        return

    # ---- Parse extract_training_features.py MODEL_REGISTRY (different shape: tuples) ----
    try:
        ext_registry = parse_dict_assign(extract_src, "MODEL_REGISTRY")
        suite.add("parse extract MODEL_REGISTRY", True, f"{len(ext_registry)} models")
    except Exception as e:
        suite.add("parse extract MODEL_REGISTRY", False, str(e))
        return

    try:
        ext_probe_cands = parse_dict_assign(extract_src, "PROBE_CANDIDATES")
        suite.add("parse extract PROBE_CANDIDATES", True, f"{len(ext_probe_cands)} models")
    except Exception as e:
        suite.add("parse extract PROBE_CANDIDATES", False, str(e))
        return

    # ---- Validate every MODEL_REGISTRY entry has required fields ----
    required_fields = {"hf_id", "loader", "input_kind", "probe_candidates"}
    required_sites = {"enc_out", "post_proj", "llm_8", "llm_16"}
    for mkey, spec in registry.items():
        missing = required_fields - set(spec.keys())
        if missing:
            suite.add(f"registry[{mkey}] required fields", False, f"missing {missing}")
            continue
        suite.add(f"registry[{mkey}] required fields", True)

        # Probe candidates must be a list with each element having all 4 sites
        cands = spec["probe_candidates"]
        if not isinstance(cands, list) or not cands:
            suite.add(f"registry[{mkey}] probe_candidates", False, "not a non-empty list")
            continue
        for i, c in enumerate(cands):
            cmissing = required_sites - set(c.keys())
            if cmissing:
                suite.add(f"registry[{mkey}].probe_candidates[{i}]", False,
                          f"missing sites {cmissing}")
                break
        else:
            suite.add(f"registry[{mkey}] probe_candidates ({len(cands)})", True)

    # ---- Cross-check: every loader name in registry exists in _LOADERS ----
    registry_loaders = {spec["loader"] for spec in registry.values()}
    loader_names = set(loaders.keys())
    missing_loaders = registry_loaders - loader_names
    if missing_loaders:
        suite.add("loader dispatch coverage", False,
                  f"loaders missing from _LOADERS: {missing_loaders}")
    else:
        suite.add("loader dispatch coverage", True,
                  f"{len(registry_loaders)} loaders all dispatchable")

    # ---- Cross-check: every loader name maps to a function that exists ----
    func_names = collect_function_names(probe_src)
    for ln, fn_name in loaders.items():
        if fn_name not in func_names and fn_name != "<expr>":
            suite.add(f"_LOADERS[{ln}] func exists", False, f"function {fn_name} not defined")
        else:
            suite.add(f"_LOADERS[{ln}] func exists", True, fn_name)

    # ---- Cross-check: every model in registry is in MODEL_COMPRESSION ----
    for mkey in registry:
        if mkey not in compression:
            suite.add(f"MODEL_COMPRESSION[{mkey}]", False, "missing entry")
        else:
            suite.add(f"MODEL_COMPRESSION[{mkey}]", True, f"= {compression[mkey]}")

    # ---- Cross-check: every model in registry is in MODEL_H3_HITS ----
    for mkey in registry:
        if mkey not in h3:
            suite.add(f"MODEL_H3_HITS[{mkey}]", False, "missing entry")
        else:
            suite.add(f"MODEL_H3_HITS[{mkey}]", True, f"= {h3[mkey]!r}")

    # ---- Cross-check: extract MODEL_REGISTRY mirrors probe MODEL_REGISTRY ----
    probe_keys = set(registry.keys())
    extract_keys = set(ext_registry.keys())
    if probe_keys != extract_keys:
        suite.add("extract MODEL_REGISTRY parity", False,
                  f"diff: probe-only={probe_keys - extract_keys}, "
                  f"extract-only={extract_keys - probe_keys}")
    else:
        suite.add("extract MODEL_REGISTRY parity", True, f"{len(probe_keys)} models")

    # ---- Cross-check: extract PROBE_CANDIDATES has every model ----
    for mkey in registry:
        if mkey not in ext_probe_cands:
            suite.add(f"extract PROBE_CANDIDATES[{mkey}]", False, "missing")
        else:
            suite.add(f"extract PROBE_CANDIDATES[{mkey}]", True,
                      f"{len(ext_probe_cands[mkey])} candidate(s)")

    # ---- ACTIVE/CACHED classification sanity ----
    classified = ACTIVE_MODELS | CACHED_MODELS | DEPRECATED_MODELS
    unclassified = set(registry.keys()) - classified
    if unclassified:
        suite.add("ACTIVE/CACHED classification", False,
                  f"unclassified models: {unclassified} (update tests/regression.py)")
    else:
        suite.add("ACTIVE/CACHED classification", True,
                  f"all {len(registry)} registry models classified")

    # ---- Cached JSONs exist on disk AND contain probe_results ----
    # Check `probe_results` (the actual data) rather than `extraction_stats`
    # — older JSONs from --probe-only resume runs have stats=0 but full results.
    for mkey in CACHED_MODELS:
        json_path = RESULTS / "week1" / f"{mkey}_quant_qual_probe.json"
        if not json_path.exists():
            suite.add(f"cached JSON: {mkey}", False, f"MISSING {json_path}")
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            n_samples = data.get("n_samples", 0)
            pr = data.get("probe_results", {}) or {}
            sites_found = set()
            for tgt in pr.values():
                if isinstance(tgt, dict):
                    sites_found.update(tgt.keys())
            required_sites = {"enc_out", "post_proj", "llm_8", "llm_16"}
            missing = required_sites - sites_found
            if missing:
                suite.add(f"cached JSON: {mkey}", False,
                          f"missing probe sites {missing}")
            elif n_samples < 50:
                suite.add(f"cached JSON: {mkey}", False,
                          f"only n_samples={n_samples} (need >=50)")
            else:
                suite.add(f"cached JSON: {mkey}", True,
                          f"n={n_samples}, all 4 sites probed")
        except Exception as e:
            suite.add(f"cached JSON: {mkey}", False, f"parse failed: {e}")

    # ---- SLURM scripts exist for ACTIVE models ----
    slurm_map = {
        "llava-onevision-7b":      "10_probe_llava_ov.sh",
        "pixtral-12b":             "11_probe_pixtral.sh",
        "phi3.5-vision":           "12_probe_phi35v.sh",
        "idefics3-8b":             "13_probe_idefics3.sh",
        "granite-vision-3.2-2b":   "14_probe_granite_vision.sh",
    }
    for mkey, script in slurm_map.items():
        sp = TURING / script
        if not sp.exists():
            suite.add(f"SLURM script: {script}", False, "MISSING")
            continue
        body = sp.read_text(encoding="utf-8", errors="replace")
        if mkey not in body:
            suite.add(f"SLURM script: {script}", False,
                      f"doesn't reference model key {mkey!r}")
            continue
        if "vla_physics_v2" not in body:
            suite.add(f"SLURM script: {script}", False,
                      "doesn't activate vla_physics_v2 env")
            continue
        suite.add(f"SLURM script: {script}", True)


# =============================================================================
# Tier 2: ACTIVE --needs transformers; verifies class import for ACTIVE models
# =============================================================================

def test_active(suite: TestSuite) -> None:
    """Test that transformers classes for ACTIVE models can be imported."""
    print()
    print(">> Tier: ACTIVE (needs `transformers` installed)")
    try:
        import transformers
        suite.add("import transformers", True, transformers.__version__)
    except ImportError as e:
        suite.add("import transformers", False, str(e))
        return

    # ACTIVE model → expected transformers class name
    active_classes = {
        "llava-onevision-7b":      "LlavaOnevisionForConditionalGeneration",
        "phi3.5-vision":           "AutoModelForCausalLM",  # via trust_remote_code
        "granite-vision-3.2-2b":   "LlavaNextForConditionalGeneration",
        "idefics3-8b":             "Idefics3ForConditionalGeneration",
    }
    for mkey, cls_name in active_classes.items():
        try:
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                suite.add(f"transformers.{cls_name}", False,
                          f"NOT in transformers (needed by {mkey})")
            else:
                suite.add(f"transformers.{cls_name}", True, f"({mkey})")
        except Exception as e:
            suite.add(f"transformers.{cls_name}", False, str(e))


# =============================================================================
# Tier 3: FULL --also tries config-only loads (small downloads)
# =============================================================================

def test_full(suite: TestSuite) -> None:
    """Try AutoConfig.from_pretrained for ACTIVE models. Downloads config.json only."""
    print()
    print(">> Tier: FULL (config-only network test, small downloads)")
    try:
        from transformers import AutoConfig
        suite.add("import AutoConfig", True)
    except ImportError as e:
        suite.add("import AutoConfig", False, str(e))
        return

    # Re-parse registry to get hf_id mappings
    probe_src = (SCRIPTS / "week1_quant_qual_probe.py").read_text(encoding="utf-8")
    registry = parse_dict_assign(probe_src, "MODEL_REGISTRY")

    for mkey in ACTIVE_MODELS:
        if mkey not in registry:
            suite.add(f"AutoConfig: {mkey}", False, "not in registry")
            continue
        hf_id = registry[mkey]["hf_id"]
        try:
            cfg = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
            mt = getattr(cfg, "model_type", "?")
            suite.add(f"AutoConfig: {mkey}", True, f"model_type={mt}")
        except Exception as e:
            suite.add(f"AutoConfig: {mkey}", False,
                      f"{type(e).__name__}: {str(e)[:100]}")


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["structural", "active", "full"],
                    default="structural",
                    help="Test tier. structural=no deps; active=needs transformers; "
                         "full=also tries config-only network loads")
    args = ap.parse_args()

    suite = TestSuite(args.tier)

    test_structural(suite)
    if args.tier in ("active", "full"):
        test_active(suite)
    if args.tier == "full":
        test_full(suite)

    return suite.report()


if __name__ == "__main__":
    sys.exit(main())
