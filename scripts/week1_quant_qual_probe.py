#!/usr/bin/env python3
"""
Week 1 killer experiment — quantitative vs qualitative physics localization.

This script is the single highest-leverage action of the NeurIPS 2026 sprint:
it tests whether the PhysBench merger bottleneck is DOMAIN-CONDITIONAL, i.e.
specific to numerical/quantitative physical attributes (size/mass/number/
distance/temperature) rather than qualitative physics (dynamics/relationships/
scene/attribute).

Core claim under test:
    At the `post_proj` (merger) site, probe accuracy for quantitative questions
    drops MORE than for qualitative questions, compared to the `enc_out` site.
    This would refine Hidden in Plain Sight (2025) rather than contradict it:
    general visual info survives the stack (their claim), but quantitative
    physics specifically degrades at the projection.

Method:
    1. Load PhysBench val (200 samples), split quant (55) vs qual (145) via
       src.optim.classify_quantitative (sub_type gold-standard).
    2. Load Qwen3-VL-8B in 4-bit nf4 bnb + bf16 compute + SDPA attention.
    3. Register forward hooks at 4 probe sites (enc_out, post_proj, llm_8,
       llm_16). For each sample, run ONE forward pass (no .generate()) to
       populate the hooks. Pool to [batch, dim] and write to FeatureCache.
    4. Fit logistic regression with 5-fold CV at each (site x slice) to
       predict the correct answer letter (A/B/C/D). Report per-site accuracy
       for quant and qual slices.
    5. Dump results to results/week1/quant_qual_probe.json and print a table.

Resume-safe: incremental JSONL writes, FeatureCache skips already-cached IDs,
crash at sample N loses at most 1 sample.

Usage:
    # Discover Qwen3-VL-8B module paths (run first if defaults fail)
    python scripts/week1_quant_qual_probe.py --print-structure

    # Smoke test on 20 samples
    python scripts/week1_quant_qual_probe.py --max-samples 20

    # Full run (200 samples)
    python scripts/week1_quant_qual_probe.py
"""

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Apply CUDA env BEFORE importing torch.
from src.optim.vram import set_cuda_alloc_env  # noqa: E402
set_cuda_alloc_env()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from src.optim import (  # noqa: E402
    build_bnb_config,
    snapshot_vram,
    format_vram_delta,
    hard_cleanup,
    pick_attn_impl,
    inference_ctx,
    ProbeSites,
    FeatureCache,
    register_probe_hooks,
    PromptCache,
    JsonlAppender,
    resume_completed_ids,
    configure_traceback_logging,
    classify_quantitative,
    split_physbench,
)
from src.optim.features import _resolve_module, _pool  # noqa: E402

# Reuse existing project helpers for PhysBench prompt/media assembly.
from scripts.run_physbench_eval import (  # noqa: E402
    load_physbench_data,
    resolve_media_paths,
    format_question_for_vlm,
)


# ---------------------------------------------------------------------------
# Model loading (Qwen3-VL-8B in 4-bit, via src.optim).
# ---------------------------------------------------------------------------

def load_qwen3_vl_8b(model_id: str = "Qwen/Qwen3-VL-8B-Instruct"):
    """Load Qwen3-VL-8B using the src.optim stack.

    Returns (model, processor).
    """
    from transformers import AutoProcessor
    try:
        from transformers import Qwen3VLForConditionalGeneration as ModelCls
    except ImportError:
        from transformers import AutoModelForVision2Seq as ModelCls

    attn_impl = pick_attn_impl()
    print(f"Loading {model_id}")
    print(f"  attn_implementation={attn_impl}, quant=bnb-nf4, dtype=bf16")
    t0 = time.time()
    model = ModelCls.from_pretrained(
        model_id,
        quantization_config=build_bnb_config(load_in_4bit=True),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")
    return model, processor


# ---------------------------------------------------------------------------
# Probe site discovery — handles Qwen3-VL module-path variants robustly.
# ---------------------------------------------------------------------------

def discover_probe_sites(model: nn.Module) -> ProbeSites:
    """Return a ProbeSites config that actually resolves on this model.

    Tries several candidate path sets for Qwen3-VL-8B (HF structure varies
    slightly between transformers versions). Falls back to walking the module
    tree if none work.
    """
    # IMPORTANT: enc_out must be the LAST ViT block (pre-merger), not the
    # `visual` wrapper — the wrapper's forward includes the merger, so hooking
    # there captures post-projection features and collapses the enc_out vs
    # post_proj distinction we're testing.
    candidates = [
        # Qwen3-VL / transformers >= 4.49 nested structure (Qwen3VLForConditionalGeneration.model.visual...)
        {
            "enc_out":   "model.visual.blocks.26",        # last of 27 ViT blocks (Qwen3-VL-8B)
            "post_proj": "model.visual.merger",
            "llm_8":     "model.language_model.layers.8",
            "llm_16":    "model.language_model.layers.16",
        },
        # Some transformers versions expose the outer wrapper as the root.
        {
            "enc_out":   "visual.blocks.26",
            "post_proj": "visual.merger",
            "llm_8":     "language_model.layers.8",
            "llm_16":    "language_model.layers.16",
        },
        # Fallback for models with fewer ViT blocks — try block 23.
        {
            "enc_out":   "model.visual.blocks.23",
            "post_proj": "model.visual.merger",
            "llm_8":     "model.language_model.layers.8",
            "llm_16":    "model.language_model.layers.16",
        },
    ]
    for paths in candidates:
        try:
            for _name, path in paths.items():
                _resolve_module(model, path)
            print(f"  probe sites resolved: {paths}")
            return ProbeSites(model_name="qwen3-vl-8b", paths=paths)
        except KeyError:
            continue

    # Nothing matched — dump the top-level structure to help the user.
    print("\nERROR: could not resolve probe site paths. Model top-level children:")
    for name, _mod in model.named_children():
        print(f"  - {name}")
    print("\nRun with --print-structure for a deeper dump.")
    raise RuntimeError("probe site discovery failed")


def print_module_structure(model: nn.Module, max_depth: int = 4) -> None:
    """Print the model module tree up to max_depth. Used for one-shot discovery."""
    print("\n=== Model module tree ===")
    for name, module in model.named_modules():
        depth = name.count(".")
        if depth > max_depth:
            continue
        indent = "  " * depth
        cls = type(module).__name__
        if depth == 0:
            print(f"{cls}")
        else:
            short = name.split(".")[-1]
            print(f"{indent}{short}: {cls}")


# ---------------------------------------------------------------------------
# Per-sample forward pass + feature capture.
# ---------------------------------------------------------------------------

def build_inputs(processor, messages: list) -> dict:
    """Apply chat template + vision preprocessing; return model-ready inputs."""
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs


def forward_and_capture(
    model: nn.Module,
    inputs: dict,
    captured: Dict[str, torch.Tensor],
) -> None:
    """Run a single forward pass. Hooks populate `captured` in place."""
    # Move to model device.
    device = next(model.parameters()).device
    inputs_on_dev = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    captured.clear()
    with inference_ctx():
        _ = model(**inputs_on_dev)
    # Free.
    del inputs_on_dev


# ---------------------------------------------------------------------------
# Probing — logistic regression with 5-fold CV on each (site, slice).
# ---------------------------------------------------------------------------

def fit_probe(
    features: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    max_iter: int = 2000,
    C: float = 0.1,
    min_n: int = 5,
) -> Dict:
    """Fit multinomial logistic regression with stratified K-fold CV.

    Regularization (C=0.1) is stronger than the sklearn default (C=1.0)
    because we're fitting a 4-way classifier on high-dim features
    (1152 or 4096) with small sample sizes (55 quant, 145 qual). Without
    regularization, the probe overfits catastrophically and reports chance
    performance.

    Fallback strategy for small slices:
        n < min_n (default 5) → skip with reason.
        n < 10 → single stratified train/test split (80/20) instead of CV.
        n >= 10 → full StratifiedKFold CV with n_splits capped at min_class.

    Returns dict with mean_acc, std_acc, folds, n, chance, method, reason.
    'reason' is None on success, otherwise a one-line explanation of why
    the probe couldn't run (for the final table).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.preprocessing import StandardScaler
    from collections import Counter

    def _skip(reason: str, n: int) -> Dict:
        return {
            "mean_acc": None, "std_acc": None, "folds": [],
            "n": int(n), "chance": None, "method": None, "reason": reason,
        }

    # Filter out samples with empty / missing labels — PhysBench val has at
    # least one such orphan that would otherwise make min_class=1 and block
    # stratified splitting.
    valid_mask = np.array([bool(str(lab).strip()) for lab in labels])
    if not valid_mask.all():
        features = features[valid_mask]
        labels = labels[valid_mask]

    # Drop singleton classes (letters appearing exactly once) — they can't
    # be stratified and we'd rather probe on n-1 samples than skip entirely.
    # This can happen on small slices (55 quant samples / 4 classes has a
    # realistic chance of one class getting only 1 sample).
    label_counts = Counter(labels.tolist())
    singleton_labels = {k for k, v in label_counts.items() if v < 2}
    if singleton_labels:
        keep = np.array([lab not in singleton_labels for lab in labels])
        features = features[keep]
        labels = labels[keep]

    n = len(labels)
    unique = np.unique(labels)
    n_classes = len(unique)

    if n < min_n:
        return _skip(f"n={n} < min_n={min_n} after filtering", n)
    if n_classes < 2:
        return _skip(f"single-class slice after filtering (only '{unique[0]}')", n)

    class_counts = Counter(labels.tolist())
    min_class = min(class_counts.values())
    chance = max(class_counts.values()) / n

    def _fit(X_train, X_test, y_train, y_test) -> float:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        clf = LogisticRegression(max_iter=max_iter, C=C, solver="lbfgs")
        clf.fit(X_train_s, y_train)
        return float(clf.score(X_test_s, y_test))

    # Tier 1: CV on n >= 10 with enough per-class samples for stratified split.
    if n >= 10 and min_class >= 2:
        k = min(n_splits, min_class)
        if k >= 2:
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            accs = []
            for tr, te in skf.split(features, labels):
                accs.append(_fit(features[tr], features[te], labels[tr], labels[te]))
            accs = np.array(accs)
            return {
                "mean_acc": float(accs.mean()),
                "std_acc":  float(accs.std()),
                "folds":    [float(a) for a in accs],
                "n":        int(n),
                "chance":   float(chance),
                "method":   f"stratified-{k}fold-cv",
                "reason":   None,
            }

    # Tier 2: single stratified train/test split for very small slices.
    if min_class >= 2:
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                features, labels, test_size=0.3, random_state=42, stratify=labels,
            )
            acc = _fit(X_tr, X_te, y_tr, y_te)
            return {
                "mean_acc": float(acc),
                "std_acc":  0.0,
                "folds":    [float(acc)],
                "n":        int(n),
                "chance":   float(chance),
                "method":   "single-split-70-30-stratified",
                "reason":   None,
            }
        except ValueError as e:
            return _skip(f"stratified split failed: {e}", n)

    return _skip(
        f"insufficient per-class samples (min_class={min_class} < 2)", n,
    )


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--data-dir", default="data/physbench", type=Path)
    ap.add_argument("--output-dir", default="results/week1", type=Path)
    ap.add_argument("--cache-dir", default="cache/week1", type=Path)
    ap.add_argument("--log-dir", default="logs", type=Path)
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Cap sample count for smoke testing.")
    ap.add_argument("--print-structure", action="store_true",
                    help="Load model, print module tree, and exit.")
    ap.add_argument("--skip-probe", action="store_true",
                    help="Only extract features, skip the probing step.")
    ap.add_argument("--features-only", action="store_true",
                    help="Alias for --skip-probe.")
    ap.add_argument("--fresh", action="store_true",
                    help="Delete the JSONL log (NOT the feature cache) before starting. "
                         "Use when the log has stale error entries from a failed run "
                         "that are blocking retry of those samples.")
    ap.add_argument("--probe-only", action="store_true",
                    help="Skip feature extraction, only run probing on whatever is "
                         "currently in the FeatureCache. Useful for rerunning the "
                         "probe step with different hyperparameters.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    logger = configure_traceback_logging(args.log_dir, "week1_quant_qual")
    logger.info("=" * 70)
    logger.info("Week 1 experiment: quantitative vs qualitative physics localization")
    logger.info("=" * 70)

    # ---- Load PhysBench val + split ----
    samples = load_physbench_data(str(args.data_dir), split="val",
                                  max_samples=args.max_samples)
    for s in samples:
        s.setdefault("sample_id", f"val_{s.get('idx', '?')}")
    quant_samples, qual_samples = split_physbench(samples)
    logger.info(f"PhysBench val: {len(samples)} total, "
                f"{len(quant_samples)} quantitative, {len(qual_samples)} qualitative")

    # ---- Fresh start handling + dirty-state detection ----
    jsonl_path = args.output_dir / "feature_extraction.jsonl"
    if args.fresh and jsonl_path.exists():
        logger.info(f"--fresh: removing stale JSONL at {jsonl_path}")
        jsonl_path.unlink()

    # Peek at current state to warn the user about inconsistencies.
    cache_preview = FeatureCache(args.cache_dir / "features", "qwen3-vl-8b", "val")
    cache_ids = cache_preview.completed_ids()
    del cache_preview

    if jsonl_path.exists():
        ok_ids = resume_completed_ids(jsonl_path)  # filters by status=ok now
        # Count all entries for the dirty-state warning.
        import json as _json
        total_entries = 0
        error_entries = 0
        with open(jsonl_path, "r", encoding="utf-8") as _fh:
            for _line in _fh:
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    _obj = _json.loads(_line)
                except Exception:
                    continue
                total_entries += 1
                if _obj.get("status") and _obj.get("status") != "ok":
                    error_entries += 1
        logger.info(f"JSONL state: {total_entries} entries "
                    f"({len(ok_ids)} ok, {error_entries} non-ok)")
        if error_entries > 0 and not args.fresh:
            logger.warning(
                f"JSONL has {error_entries} non-ok entries from a previous run. "
                f"These will be RETRIED (resume now filters by status=ok). "
                f"Pass --fresh to wipe the log entirely."
            )
        if len(ok_ids) != len(cache_ids):
            logger.warning(
                f"JSONL ok count ({len(ok_ids)}) does not match FeatureCache "
                f"cached count ({len(cache_ids)}). Will trust the FeatureCache."
            )

    # ---- Load model (skipped in --probe-only mode) ----
    if args.probe_only:
        logger.info("--probe-only: skipping Qwen3-VL-8B load entirely "
                    "(probing runs on cached .npy files only)")
        model = None
        processor = None
        # Build a synthetic ProbeSites from whatever the cache contains so the
        # downstream code has a consistent `sites.paths` to iterate over.
        cache_preview2 = FeatureCache(args.cache_dir / "features", "qwen3-vl-8b", "val")
        cache_sites = list(cache_preview2._index.get("dims", {}).keys())
        del cache_preview2
        if not cache_sites:
            logger.error("--probe-only requires cached features; cache is empty. "
                         "Run without --probe-only first to populate the cache.")
            return 2
        sites = ProbeSites(
            model_name="qwen3-vl-8b",
            paths={name: f"cached::{name}" for name in cache_sites},
        )
        captured, handles = {}, []
    else:
        before_vram = snapshot_vram()
        model, processor = load_qwen3_vl_8b(args.model_id)
        after_vram = snapshot_vram()
        logger.info(format_vram_delta(before_vram, after_vram))

        if args.print_structure:
            print_module_structure(model, max_depth=3)
            return 0

        # ---- Discover + register probe sites ----
        sites = discover_probe_sites(model)
        captured, handles = register_probe_hooks(model, sites)

    # ---- Feature extraction with resume ----
    # Resume is driven by the FeatureCache (source of truth for "has features").
    # The JSONL is an audit log; we only use it to warn about dirty state.
    cache = FeatureCache(args.cache_dir / "features", "qwen3-vl-8b", "val")
    cached_id_set = cache.completed_ids()
    logger.info(f"Resume: FeatureCache has {len(cached_id_set)} cached samples")

    results: Dict[str, Dict] = {}
    site_feats: Dict[str, List[np.ndarray]] = {site: [] for site in sites.paths}
    # Multiple probe targets — evaluated independently at every site.
    # 'answer':    predict A/B/C/D letter (weak signal, position-of-answer).
    # 'task_type': predict dynamics/property/relationships/scene
    #              (the PhysBench physics category — the real paper claim).
    # 'sub_type':  predict fine-grained sub_type (19 classes, sparse per class).
    site_target_labels: Dict[str, Dict[str, List[str]]] = {
        site: {"answer": [], "task_type": [], "sub_type": []}
        for site in sites.paths
    }
    site_slices: Dict[str, List[str]] = {site: [] for site in sites.paths}

    # Replay cached samples into memory for probing. FeatureCache uses
    # per-sample files on disk — load_site returns a stacked [N, D] ndarray
    # in the order given by index.json sample_ids.
    cached_ids = list(cache.completed_ids())
    if cached_ids:
        logger.info(f"Loading {len(cached_ids)} cached feature vectors from disk")
        id_to_answer    = {s["sample_id"]: s.get("answer", "") for s in samples}
        id_to_task      = {s["sample_id"]: s.get("task_type", "") for s in samples}
        id_to_sub       = {s["sample_id"]: s.get("sub_type", "") for s in samples}
        id_to_slice     = {s["sample_id"]: classify_quantitative(s) for s in samples}
        index_order = list(cache._index["sample_ids"])
        for site in sites.paths:
            stacked = cache.load_site(site)  # shape [N, dim]
            for i, sid in enumerate(index_order):
                if sid in id_to_answer:
                    site_feats[site].append(np.asarray(stacked[i]))
                    site_target_labels[site]["answer"].append(id_to_answer[sid])
                    site_target_labels[site]["task_type"].append(id_to_task[sid])
                    site_target_labels[site]["sub_type"].append(id_to_sub[sid])
                    site_slices[site].append(id_to_slice[sid])
            del stacked

    total = len(samples)
    processed = 0
    skipped = 0
    errors = 0
    t_start = time.time()

    # --probe-only: bypass the extraction loop entirely, remove hooks, go
    # straight to probing with whatever is already cached.
    if args.probe_only:
        logger.info("--probe-only: skipping feature extraction, "
                    "probing only against already-cached features.")
        for h in handles:
            h.remove()
        handles = []
    else:
        try:
            with JsonlAppender(jsonl_path) as out:
                for i, item in enumerate(samples):
                    sid = item["sample_id"]
                    # Only skip if FeatureCache has this sample — JSONL errors
                    # no longer block retry (resume_completed_ids filters by
                    # status=ok, and we use the FeatureCache as the single
                    # source of truth for "extracted").
                    if sid in cache.completed_ids():
                        skipped += 1
                        continue

                    media_paths = resolve_media_paths(item, str(args.data_dir))
                    if not any(p for p in media_paths):
                        logger.warning(f"{sid}: no media files resolved, skipping")
                        out.write({"sample_id": sid, "status": "no_media"})
                        errors += 1
                        continue

                    try:
                        messages, has_media = format_question_for_vlm(item, media_paths)
                        if not has_media:
                            logger.warning(f"{sid}: has_media=False, skipping")
                            out.write({"sample_id": sid, "status": "no_media_resolved"})
                            errors += 1
                            continue

                        inputs = build_inputs(processor, messages)
                        forward_and_capture(model, inputs, captured)

                        # Pull per-site features. Qwen3-VL vision blocks output
                        # [total_patches, dim] (no explicit batch dim); LLM
                        # layers output [batch, seq, dim]. Universal: collapse
                        # all non-last dims via mean-pool to [dim], then
                        # unsqueeze to [1, dim] for the batch=1 cache write.
                        np_feats: Dict[str, np.ndarray] = {}
                        for site in sites.paths:
                            if site not in captured:
                                raise RuntimeError(f"hook site {site} did not fire")
                            t = captured[site]
                            if t.ndim == 0:
                                raise RuntimeError(f"site {site}: scalar tensor")
                            if t.ndim == 1:
                                pooled = t
                            else:
                                reduce_dims = tuple(range(t.ndim - 1))
                                pooled = t.mean(dim=reduce_dims)
                            np_feats[site] = pooled.unsqueeze(0).numpy().astype(np.float32)

                        cache.append_batch([sid], np_feats)

                        ans_label  = item.get("answer", "")
                        task_label = item.get("task_type", "")
                        sub_label  = item.get("sub_type", "")
                        slice_name = classify_quantitative(item)
                        for site in sites.paths:
                            site_feats[site].append(np_feats[site][0])
                            site_target_labels[site]["answer"].append(ans_label)
                            site_target_labels[site]["task_type"].append(task_label)
                            site_target_labels[site]["sub_type"].append(sub_label)
                            site_slices[site].append(slice_name)

                        out.write({
                            "sample_id": sid,
                            "status":    "ok",
                            "slice":     slice_name,
                            "answer":    label,
                            "task_type": item.get("task_type"),
                            "sub_type":  item.get("sub_type"),
                            "dims":      {k: int(v.shape[-1]) for k, v in np_feats.items()},
                        })
                        processed += 1

                        if processed % 10 == 0 or processed == 1:
                            elapsed = time.time() - t_start
                            rate = processed / max(elapsed, 1e-6)
                            remaining = (total - i - 1) / max(rate, 1e-6)
                            vram = snapshot_vram()
                            logger.info(
                                f"  [{i+1}/{total}] ok={processed} skip={skipped} err={errors} "
                                f"| {rate:.2f} q/s | ETA {remaining/60:.1f}min "
                                f"| VRAM {vram.allocated_gb:.1f}GB"
                            )

                    except Exception as e:
                        errors += 1
                        logger.error(f"{sid}: {type(e).__name__}: {e}")
                        logger.debug(traceback.format_exc())
                        out.write({"sample_id": sid, "status": "error",
                                   "error": f"{type(e).__name__}: {e}"})
                        torch.cuda.empty_cache()

        finally:
            for h in handles:
                h.remove()

        elapsed = time.time() - t_start
        logger.info(f"Extraction complete: processed={processed} skipped={skipped} "
                    f"errors={errors} in {elapsed/60:.1f} min")

    # ---- Free model before probing to avoid CUDA OOM during sklearn ----
    if model is not None:
        final_vram = snapshot_vram()
        logger.info(f"Pre-cleanup {final_vram}")
        hard_cleanup(model, processor)
        post_vram = snapshot_vram()
        logger.info(format_vram_delta(final_vram, post_vram))

    if args.skip_probe or args.features_only:
        logger.info("Skipping probing step (--skip-probe).")
        return 0

    # ---- Cache sanity check before probing ----
    logger.info("\n" + "=" * 70)
    logger.info("Cache sanity check")
    logger.info("=" * 70)
    sanity_failures: List[str] = []
    for site in sites.paths:
        if not site_feats[site]:
            sanity_failures.append(f"  {site}: empty")
            continue
        feats = np.stack(site_feats[site])
        n_nan = int(np.isnan(feats).sum())
        n_inf = int(np.isinf(feats).sum())
        n_zero_rows = int((feats == 0).all(axis=1).sum())
        logger.info(
            f"  {site}: n={feats.shape[0]} dim={feats.shape[1]} "
            f"mean={feats.mean():.4f} std={feats.std():.4f} "
            f"nan={n_nan} inf={n_inf} zero_rows={n_zero_rows}"
        )
        if n_nan > 0 or n_inf > 0:
            sanity_failures.append(f"  {site}: NaN/Inf contamination")
        if feats.std() < 1e-6:
            sanity_failures.append(f"  {site}: degenerate (std ≈ 0)")
    if sanity_failures:
        logger.error("Cache sanity check FAILED:")
        for f in sanity_failures:
            logger.error(f)
        logger.error("Proceeding with probing anyway — results may be meaningless.")

    # ---- Probing: fit LogReg for each (target, site, slice) ----
    logger.info("\n" + "=" * 70)
    logger.info("Probing: fit LogReg for each target x site x {all, quant, qual}")
    logger.info("=" * 70)

    PROBE_TARGETS = ["answer", "task_type", "sub_type"]

    # Structure: probe_results[target][site] = {"all": ..., "quantitative": ..., "qualitative": ..., meta}
    probe_results: Dict[str, Dict[str, Dict]] = {t: {} for t in PROBE_TARGETS}

    for site in sites.paths:
        if not site_feats[site]:
            for t in PROBE_TARGETS:
                probe_results[t][site] = {
                    "all": None, "quantitative": None, "qualitative": None,
                    "n_total": 0, "n_quant": 0, "n_qual": 0, "feature_dim": None,
                }
            continue
        feats = np.stack(site_feats[site])
        slices = np.array(site_slices[site])
        quant_mask = slices == "quantitative"
        qual_mask = slices == "qualitative"

        logger.info(f"  {site}: n_total={len(feats)} n_quant={quant_mask.sum()} "
                    f"n_qual={qual_mask.sum()} feat_dim={feats.shape[1]}")

        for target in PROBE_TARGETS:
            labels = np.array(site_target_labels[site][target])
            all_probe = fit_probe(feats, labels)
            quant_probe = fit_probe(feats[quant_mask], labels[quant_mask])
            qual_probe = fit_probe(feats[qual_mask], labels[qual_mask])
            probe_results[target][site] = {
                "all":          all_probe,
                "quantitative": quant_probe,
                "qualitative":  qual_probe,
                "n_total":      int(len(feats)),
                "n_quant":      int(quant_mask.sum()),
                "n_qual":       int(qual_mask.sum()),
                "feature_dim":  int(feats.shape[1]),
            }

    # Save full results.
    results_path = args.output_dir / "quant_qual_probe.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model_id,
            "split": "val",
            "n_samples": len(samples),
            "n_quant": len(quant_samples),
            "n_qual": len(qual_samples),
            "targets": PROBE_TARGETS,
            "probe_results": probe_results,
            "extraction_stats": {
                "processed": processed,
                "skipped": skipped,
                "errors": errors,
                "probe_only": args.probe_only,
            },
        }, f, indent=2)
    logger.info(f"Results written to {results_path}")

    # ------------------------------------------------------------------
    # Pretty-print the comparison table — one block per probe target.
    # ------------------------------------------------------------------
    def _fmt(probe: Optional[Dict]) -> str:
        if probe is None:
            return "skip (no data)"
        if probe.get("mean_acc") is None:
            reason = probe.get("reason") or "unknown"
            return f"skip ({reason[:40]})"
        acc = probe["mean_acc"]
        std = probe.get("std_acc") or 0.0
        return f"{acc:.3f}+/-{std:.3f}"

    site_order = ["enc_out", "post_proj", "llm_8", "llm_16"]

    # Target-specific tag lines explaining what a positive signal looks like.
    TARGET_INTERPRETATION = {
        "answer":    "predicts the correct A/B/C/D letter. Weak signal expected "
                     "(letter position is near-random by PhysBench construction).",
        "task_type": "predicts dynamics/property/relationships/scene. STRONG signal "
                     "expected if features encode physics category at all.",
        "sub_type":  "predicts fine-grained sub_type (19 classes). Stress test for "
                     "high-granularity physics concept encoding.",
    }

    for target in PROBE_TARGETS:
        target_results = probe_results[target]
        print("\n" + "=" * 100)
        print(f" WEEK 1 RESULTS - target='{target}'")
        print(f" {TARGET_INTERPRETATION[target]}")
        print("=" * 100)
        print(f"{'site':<12} {'n_tot':>6} {'n_q':>5} {'n_ql':>5}  "
              f"{'all_acc':<20} {'quant_acc':<22} {'qual_acc':<22} {'d(qual-quant)':>14}")
        print("-" * 100)
        for site in site_order:
            if site not in target_results:
                continue
            pr = target_results[site]
            all_p = pr.get("all")
            q_p = pr.get("quantitative")
            l_p = pr.get("qualitative")
            delta = None
            if q_p and l_p and q_p.get("mean_acc") is not None and l_p.get("mean_acc") is not None:
                delta = l_p["mean_acc"] - q_p["mean_acc"]
            delta_str = f"{delta:+.3f}" if delta is not None else "n/a"
            print(
                f"{site:<12} {pr['n_total']:>6} {pr['n_quant']:>5} {pr['n_qual']:>5}  "
                f"{_fmt(all_p):<20} {_fmt(q_p):<22} {_fmt(l_p):<22} {delta_str:>14}"
            )
        print("=" * 100)

    print("\nKEY:")
    print("  all_acc = probe accuracy on the full slice (ignoring quant/qual split)")
    print("  d(qual-quant) = qual_acc - quant_acc.")
    print("  H3 (domain-conditional bottleneck): on target='task_type' or 'sub_type',")
    print("  quant_acc should drop MORE than qual_acc across enc_out -> post_proj,")
    print("  making d(qual-quant) grow POSITIVE from enc_out to post_proj.")
    print()

    # Skip report.
    skips = []
    for target in PROBE_TARGETS:
        for site in site_order:
            if site not in probe_results[target]:
                continue
            for slot in ("all", "quantitative", "qualitative"):
                p = probe_results[target][site].get(slot)
                if p and p.get("reason"):
                    skips.append(f"  {target}.{site}.{slot}: n={p.get('n')} -- {p['reason']}")
    if skips:
        print("Skipped probe slots (with reasons):")
        for s in skips[:20]:  # cap to avoid wall of text
            print(s)
        if len(skips) > 20:
            print(f"  ... and {len(skips) - 20} more skips (see {results_path})")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
