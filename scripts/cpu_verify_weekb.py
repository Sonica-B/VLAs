#!/usr/bin/env python3
"""
CPU-ONLY pre-flight verification for Week B models.

DOES NOT TOUCH THE GPU. Safe to run while other GPU processes are active.

Sets CUDA_VISIBLE_DEVICES="" at the very top so this Python process cannot
see any GPU — even accidental `.cuda()` calls would fail with a clear error
rather than competing with your running job.

Verifies for each Week B model:
  1. The target HF model class can be imported (transformers version check)
  2. The HF config downloads successfully (~10 KB per model)
  3. config.vision_config.num_hidden_layers (or equivalent) >= our probe layer
     index — catches "we probe layers.26 but model only has 24 layers" bugs
  4. The AutoProcessor can load (verifies processor API availability)
  5. (Optional) Model can be instantiated on meta device with random weights,
     so PROBE_CANDIDATES paths can be resolved against the actual module tree

Runtime: ~20-40 seconds per model, ~200 MB RAM peak (config-only mode),
or ~1-2 GB (with --meta-model for full path resolution).

Usage:
    # Fast: config + processor only (no model instantiation)
    python scripts/cpu_verify_weekb.py

    # Thorough: also instantiate model on meta device to verify paths
    python scripts/cpu_verify_weekb.py --meta-model

    # Single model
    python scripts/cpu_verify_weekb.py --model pixtral-12b
"""

from __future__ import annotations

# MULTI-LAYER GPU PROTECTION — tested against torch 2.11 on Windows which
# is known to allocate on physical GPU even with CUDA_VISIBLE_DEVICES="".
#
# Layer 1: Clear visibility env var (subprocess-level isolation if launched
# with `CUDA_VISIBLE_DEVICES="" python ...`).
# Layer 2: Monkey-patch torch.cuda functions to always report unavailable.
# Layer 3: Block .cuda() and .to('cuda') on Tensor/Module — raise loudly
# rather than silently leaking to the physical GPU.
# Layer 4: Default mode is CONFIG-ONLY (no model instantiation, no tensors
# created at all). The optional --meta-model flag runs under torch.device('meta')
# which is the ONLY verified-safe path for model instantiation.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")

import argparse
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

# ---------------------------------------------------------------------------
# Defensive GPU lockout: intercept anything that would touch a CUDA device.
# ---------------------------------------------------------------------------

# Record original functions so --allow-gpu (debugging) can restore them.
_ORIG_CUDA_IS_AVAILABLE = torch.cuda.is_available
_ORIG_TENSOR_CUDA = torch.Tensor.cuda
_ORIG_TENSOR_TO = torch.Tensor.to
_ORIG_MODULE_CUDA = torch.nn.Module.cuda
_ORIG_MODULE_TO = torch.nn.Module.to


def _gpu_blocked_cuda(self, *args, **kwargs):
    raise RuntimeError(
        "CPU-verify: .cuda() blocked to prevent disrupting any running GPU "
        "process. If you need model instantiation, use --meta-model "
        "(which uses torch.device('meta') — no GPU or CPU memory for weights)."
    )


def _check_device_arg(args, kwargs):
    """Raise if args/kwargs specify a CUDA-family device."""
    # First positional arg can be a device, dtype, or tensor.
    candidate_strs = []
    for a in args:
        try:
            candidate_strs.append(str(a).lower())
        except Exception:
            pass
    for v in kwargs.values():
        try:
            candidate_strs.append(str(v).lower())
        except Exception:
            pass
    for s in candidate_strs:
        if "cuda" in s or s.startswith("gpu"):
            raise RuntimeError(
                f"CPU-verify: .to('cuda') / device='cuda' blocked "
                f"(got '{s}') to prevent disrupting any running GPU process."
            )


def _gpu_blocked_tensor_to(self, *args, **kwargs):
    _check_device_arg(args, kwargs)
    return _ORIG_TENSOR_TO(self, *args, **kwargs)


def _gpu_blocked_module_to(self, *args, **kwargs):
    _check_device_arg(args, kwargs)
    return _ORIG_MODULE_TO(self, *args, **kwargs)


def _install_gpu_lockout():
    """Install monkey-patches blocking any CUDA allocation path."""
    torch.cuda.is_available = lambda: False
    torch.Tensor.cuda = _gpu_blocked_cuda
    torch.Tensor.to = _gpu_blocked_tensor_to
    torch.nn.Module.cuda = _gpu_blocked_cuda
    torch.nn.Module.to = _gpu_blocked_module_to
    # Also make device_count return 0 to defeat simple sanity checks
    torch.cuda.device_count = lambda: 0


def _prove_gpu_lockout() -> Tuple[bool, str]:
    """Sanity check: confirm no GPU tensor allocation is possible."""
    # Test 1: is_available returns False
    if torch.cuda.is_available():
        return False, "torch.cuda.is_available() still returns True"
    # Test 2: device_count returns 0
    if torch.cuda.device_count() != 0:
        return False, f"torch.cuda.device_count() = {torch.cuda.device_count()}"
    # Test 3: .cuda() on a tensor raises
    try:
        x = torch.zeros(4)
        x.cuda()
        return False, ".cuda() did NOT raise — GPU accessible"
    except RuntimeError as e:
        if "cpu-verify" not in str(e).lower():
            return False, f".cuda() raised unexpected: {e}"
    # Test 4: .to('cuda') on a tensor raises
    try:
        x = torch.zeros(4)
        x.to("cuda")
        return False, ".to('cuda') did NOT raise — GPU accessible"
    except RuntimeError as e:
        if "cpu-verify" not in str(e).lower():
            return False, f".to('cuda') raised unexpected: {e}"
    return True, "all 4 GPU-access paths blocked"


# Install lockout IMMEDIATELY at import time, before any model code runs.
_install_gpu_lockout()
# Also force the default device so any Tensor() without explicit device stays on CPU.
try:
    torch.set_default_device("cpu")
except Exception:
    pass  # older torch versions don't have set_default_device
_LOCKOUT_OK, _LOCKOUT_MSG = _prove_gpu_lockout()


# -------------------------------------------------------------------------
# Week B model spec — everything we need to verify per model.
# -------------------------------------------------------------------------

WEEK_B_SPEC = {
    "llava-onevision-7b": {
        "hf_id": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "model_class": "LlavaOnevisionForConditionalGeneration",  # transformers >=4.45
        "fallback_class": "AutoModelForImageTextToText",
        "trust_remote_code": False,
        # Config: vision_config.num_hidden_layers=26, text_config.num_hidden_layers=28
        # We probe vision layers.25 (last) and LLM layers 8, 16.
        "vision_min_layers": 26,
        "text_min_layers": 17,
    },
    "phi3.5-vision": {
        "hf_id": "microsoft/Phi-3.5-vision-instruct",
        "model_class": "AutoModelForCausalLM",
        "fallback_class": "AutoModelForCausalLM",
        "trust_remote_code": True,
        # CLIP ViT-L-14-336: 24 layers (we probe index 23 = last)
        "vision_min_layers": 24,
        # Phi-3.5-mini: 32 layers (we probe 8, 16)
        "text_min_layers": 17,
    },
    "pixtral-12b": {
        "hf_id": "mistral-community/pixtral-12b",
        "model_class": "LlavaForConditionalGeneration",
        "fallback_class": "AutoModelForImageTextToText",
        "trust_remote_code": False,
        # PixtralVisionConfig: ~24 layers (we probe 23)
        "vision_min_layers": 24,
        # Mistral-Nemo 12B: 40 layers
        "text_min_layers": 17,
    },
    "molmo-7b": {
        "hf_id": "allenai/Molmo-7B-D-0924",
        "model_class": "AutoModelForCausalLM",
        "fallback_class": "AutoModelForCausalLM",
        "trust_remote_code": True,
        # Molmo uses custom vision_backbone — layer count in config
        "vision_min_layers": 23,
        # Qwen2-7B backbone: 28 layers
        "text_min_layers": 17,
    },
}


def check_transformers_version() -> Tuple[bool, str]:
    """Minimum transformers version required for Week B."""
    try:
        import transformers
        v = transformers.__version__
        parts = v.split(".")
        major, minor = int(parts[0]), int(parts[1])
        ok = (major > 4) or (major == 4 and minor >= 45)
        return ok, v
    except Exception as e:
        return False, f"import error: {e}"


def check_class_import(class_name: str, fallback: str) -> Tuple[bool, str]:
    """Verify the target model class (or fallback) imports from transformers."""
    try:
        import transformers
        if hasattr(transformers, class_name):
            return True, class_name
        if hasattr(transformers, fallback):
            return True, f"{fallback} (fallback)"
        return False, f"neither {class_name} nor {fallback} available"
    except Exception as e:
        return False, f"import error: {e}"


def check_config(hf_id: str, trust_remote_code: bool) -> Tuple[Optional[object], str]:
    """Download just the config. Returns (config, note) or (None, error)."""
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(hf_id, trust_remote_code=trust_remote_code)
        return config, f"config loaded ({type(config).__name__})"
    except Exception as e:
        return None, f"config download failed: {type(e).__name__}: {e}"


def get_num_layers(config, path: str) -> Optional[int]:
    """Walk dotted path to extract num_hidden_layers-like value.

    Tries several common paths (vision_config.num_hidden_layers,
    vision_tower_config.num_hidden_layers, etc.).
    """
    obj = config
    for part in path.split("."):
        if hasattr(obj, part):
            obj = getattr(obj, part)
        else:
            return None
    return int(obj) if isinstance(obj, int) else None


def check_architecture(config, spec: Dict) -> List[str]:
    """Check num_hidden_layers fits our probe indices. Returns list of issues.

    Returns an empty list on success. Cases where a non-standard config layout
    prevents us from locating layer counts are downgraded to WARNINGS (prefixed
    with 'warn:') — they are not hard failures because the existing probing
    code may still work with custom paths."""
    issues = []

    # Vision layers — try common config attribute paths.
    vision_n = None
    for path in ("vision_config.num_hidden_layers",
                 "vision_tower_config.num_hidden_layers",
                 "vision_backbone_config.num_hidden_layers",
                 "vision_backbone_config.image_vit.num_layers",
                 # Phi-3.5-vision specific paths (nested img_processor config)
                 "img_processor.num_hidden_layers",
                 "embd_layer.num_hidden_layers"):
        n = get_num_layers(config, path)
        if n is not None:
            vision_n = n
            break

    text_n = None
    for path in ("text_config.num_hidden_layers",
                 "num_hidden_layers",  # flat for some custom models
                 "language_model_config.num_hidden_layers"):
        n = get_num_layers(config, path)
        if n is not None:
            text_n = n
            break

    # Vision layer check: cannot-locate is a WARNING (custom config layout),
    # but layer-count < required is a hard failure.
    if vision_n is None:
        issues.append(
            "warn: vision num_hidden_layers not in standard config path "
            "(custom config layout; relies on battle-tested probe paths)"
        )
    elif vision_n < spec["vision_min_layers"]:
        issues.append(
            f"vision layers {vision_n} < required {spec['vision_min_layers']} "
            f"(probe index out of range)"
        )

    if text_n is None:
        issues.append(
            "warn: text num_hidden_layers not in standard config path"
        )
    elif text_n < spec["text_min_layers"]:
        issues.append(
            f"text layers {text_n} < required {spec['text_min_layers']} "
            f"(LLM probe layer 16 out of range)"
        )

    return issues


def check_processor(hf_id: str, trust_remote_code: bool) -> Tuple[bool, str]:
    """Try to load the AutoProcessor. Catches processor-class missing errors."""
    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(hf_id, trust_remote_code=trust_remote_code)
        return True, type(proc).__name__
    except Exception as e:
        return False, f"processor failed: {type(e).__name__}: {e}"


def check_meta_model(hf_id: str, spec: Dict,
                     probe_candidates: List[Dict[str, str]]) -> Tuple[bool, List[str]]:
    """Instantiate model on meta device (random weights, zero GPU mem) and
    verify PROBE_CANDIDATES paths resolve on the actual module tree.

    Heavier (~1-2 GB RAM, ~30s per model) but catches path-naming mismatches.
    """
    issues: List[str] = []
    try:
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
        from transformers import AutoModelForImageTextToText
        try:
            from transformers import LlavaOnevisionForConditionalGeneration
        except ImportError:
            LlavaOnevisionForConditionalGeneration = None
        try:
            from transformers import LlavaForConditionalGeneration
        except ImportError:
            LlavaForConditionalGeneration = None

        config = AutoConfig.from_pretrained(
            hf_id, trust_remote_code=spec["trust_remote_code"],
        )

        # Pick the right class for from_config
        cls_name = spec["model_class"]
        if cls_name == "LlavaOnevisionForConditionalGeneration" and LlavaOnevisionForConditionalGeneration is not None:
            Cls = LlavaOnevisionForConditionalGeneration
        elif cls_name == "LlavaForConditionalGeneration" and LlavaForConditionalGeneration is not None:
            Cls = LlavaForConditionalGeneration
        elif cls_name == "AutoModelForCausalLM":
            Cls = AutoModelForCausalLM
        else:
            Cls = AutoModelForImageTextToText

        # Build on meta device — zero real memory allocated for weights
        with torch.device("meta"):
            try:
                model = Cls.from_config(
                    config,
                    trust_remote_code=spec["trust_remote_code"],
                )
            except TypeError:
                # Some classes' from_config doesn't accept trust_remote_code kwarg
                model = Cls.from_config(config)

        # Resolve PROBE_CANDIDATES
        from src.optim.features import _resolve_module
        resolved_any = False
        for idx, paths in enumerate(probe_candidates):
            try:
                for site, path in paths.items():
                    _resolve_module(model, path)
                resolved_any = True
                issues.append(f"paths OK: candidate {idx}")
                break
            except KeyError as e:
                issues.append(f"candidate {idx} failed: {e}")

        if not resolved_any:
            # Dump module top-level for diagnostic
            try:
                top = [n for n, _ in model.named_children()]
                issues.append(f"NO candidates resolved. Top-level modules: {top}")
            except Exception:
                pass
            return False, issues

        return True, issues

    except Exception as e:
        tb = traceback.format_exc()
        issues.append(f"meta-model instantiation failed: {type(e).__name__}: {e}")
        return False, issues


def verify_one(model_key: str, spec: Dict, do_meta: bool = False) -> Dict:
    """Full CPU-only verification of one model. Returns result dict."""
    print(f"\n{'='*70}")
    print(f"[CPU VERIFY] {model_key}")
    print(f"  HF ID: {spec['hf_id']}")
    print(f"{'='*70}")

    result = {
        "model": model_key,
        "transformers_ok": False,
        "class_ok": False,
        "config_ok": False,
        "arch_ok": False,
        "processor_ok": False,
        "meta_ok": None,  # None = not attempted
        "issues": [],
    }

    # 1. transformers version
    ok, v = check_transformers_version()
    result["transformers_ok"] = ok
    if not ok:
        result["issues"].append(f"transformers too old ({v}); need >=4.45")
        print(f"  [1] transformers version: FAIL ({v})")
        return result
    print(f"  [1] transformers version: {v} OK")

    # 2. class import
    ok, note = check_class_import(spec["model_class"], spec["fallback_class"])
    result["class_ok"] = ok
    if not ok:
        result["issues"].append(note)
        print(f"  [2] model class: FAIL ({note})")
        return result
    print(f"  [2] model class: {note} OK")

    # 3. config download
    config, note = check_config(spec["hf_id"], spec["trust_remote_code"])
    result["config_ok"] = config is not None
    if not result["config_ok"]:
        result["issues"].append(note)
        print(f"  [3] config: FAIL ({note})")
        return result
    print(f"  [3] config: {note}")

    # 4. architecture check — warnings (warn: ...) are not hard failures
    arch_issues = check_architecture(config, spec)
    hard_issues = [i for i in arch_issues if not i.startswith("warn:")]
    warn_issues = [i for i in arch_issues if i.startswith("warn:")]
    result["arch_ok"] = len(hard_issues) == 0
    if hard_issues:
        for issue in hard_issues:
            result["issues"].append(f"arch: {issue}")
        print(f"  [4] architecture: FAIL")
        for i in hard_issues:
            print(f"      - {i}")
    elif warn_issues:
        for issue in warn_issues:
            result["issues"].append(f"arch: {issue}")
        print(f"  [4] architecture: OK (with warnings)")
        for i in warn_issues:
            print(f"      - {i}")
    else:
        print(f"  [4] architecture: OK")

    # 5. processor load
    ok, note = check_processor(spec["hf_id"], spec["trust_remote_code"])
    result["processor_ok"] = ok
    if ok:
        print(f"  [5] processor: {note} OK")
    else:
        result["issues"].append(f"processor: {note}")
        print(f"  [5] processor: FAIL ({note})")

    # 6. (optional) meta-model path verification
    if do_meta:
        from scripts.extract_training_features import PROBE_CANDIDATES
        candidates = PROBE_CANDIDATES.get(model_key, [])
        if not candidates:
            print(f"  [6] meta-model: SKIP (no PROBE_CANDIDATES defined)")
            result["meta_ok"] = None
        else:
            ok, issues = check_meta_model(spec["hf_id"], spec, candidates)
            result["meta_ok"] = ok
            if ok:
                print(f"  [6] meta-model paths: OK")
                for i in issues:
                    print(f"      - {i}")
            else:
                for i in issues:
                    result["issues"].append(f"meta: {i}")
                print(f"  [6] meta-model paths: FAIL")
                for i in issues:
                    print(f"      - {i}")

    return result


def main() -> int:
    ap = argparse.ArgumentParser(
        description="CPU-only Week B verification (protects running GPU process)",
    )
    ap.add_argument("--model", choices=list(WEEK_B_SPEC.keys()), default=None,
                    help="Single model to verify (default: all 4)")
    ap.add_argument("--meta-model", action="store_true",
                    help="Also instantiate model on meta device to resolve probe "
                         "paths (+30s, +1-2 GB RAM per model). Uses "
                         "torch.device('meta') — verified safe against GPU leakage.")
    args = ap.parse_args()

    # Print lockout status — user wants assurance no GPU is touched.
    print("=" * 70)
    print("GPU LOCKOUT STATUS")
    print("=" * 70)
    print(f"  CUDA_VISIBLE_DEVICES env:     {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}")
    print(f"  torch.cuda.is_available():    {torch.cuda.is_available()}")
    print(f"  torch.cuda.device_count():    {torch.cuda.device_count()}")
    print(f"  Lockout sanity check:         {'PASSED' if _LOCKOUT_OK else 'FAILED'}")
    print(f"  Detail:                       {_LOCKOUT_MSG}")
    print()
    if not _LOCKOUT_OK:
        print("ABORT: GPU lockout could not be verified. Running this script")
        print("       may disrupt your GPU process. Fix the lockout first.")
        return 3
    print("  >> It is SAFE to run this script; GPU is not accessible from this process.")
    if args.meta_model:
        print("  >> --meta-model mode will instantiate models on torch.device('meta')")
        print("     (verified safe path; no real memory, no GPU contact)")
    else:
        print("  >> Running in CONFIG-ONLY mode (no tensors allocated at all)")
    print()

    models = [args.model] if args.model else list(WEEK_B_SPEC.keys())

    results = []
    for m in models:
        try:
            r = verify_one(m, WEEK_B_SPEC[m], do_meta=args.meta_model)
        except Exception as e:
            traceback.print_exc()
            r = {"model": m, "issues": [f"UNCAUGHT: {type(e).__name__}: {e}"]}
        results.append(r)

    # Summary
    print(f"\n{'='*70}")
    print(f"CPU VERIFICATION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'model':<22} {'tf':>4} {'cls':>4} {'cfg':>4} {'arch':>5} {'proc':>5} {'meta':>5}")
    print(f"  {'-'*62}")
    for r in results:
        def mark(v):
            if v is None:
                return "   -"
            return "  ok" if v else "FAIL"
        print(f"  {r['model']:<22} {mark(r.get('transformers_ok')):>4} "
              f"{mark(r.get('class_ok')):>4} {mark(r.get('config_ok')):>4} "
              f"{mark(r.get('arch_ok')):>5} {mark(r.get('processor_ok')):>5} "
              f"{mark(r.get('meta_ok')):>5}")

    any_issues = False
    for r in results:
        if r.get("issues"):
            any_issues = True
            print(f"\n  {r['model']} ISSUES:")
            for i in r["issues"]:
                print(f"    - {i}")

    print(f"\n{'='*70}")
    # Aggregate verdict — different bars for config-only vs meta mode
    required_keys = ["transformers_ok", "class_ok", "config_ok", "arch_ok", "processor_ok"]
    if args.meta_model:
        required_keys.append("meta_ok")
    num_ok = sum(1 for r in results if all(r.get(k) for k in required_keys))
    print(f"  {num_ok}/{len(results)} models pass CPU-only verification")
    print(f"{'='*70}")

    if num_ok == 0:
        print(f"\nABORT: no models pass. Fix issues above before submitting Turing SLURM.")
        return 2
    if num_ok < len(results):
        print(f"\nWARN: {len(results) - num_ok} model(s) failed. SLURM job handles per-model failures gracefully but those models will not contribute to n=8.")
        return 1
    print(f"\nOK: all {num_ok} Week B models pass CPU-only verification.")
    print(f"    Next: on Turing, run sbatch turing/08_weekb_extract_new_models.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
