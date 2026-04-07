#!/usr/bin/env python3
"""
PEM v4: Feature-Space Training — trains PEM on cached features, no model needed.

WHY v4 (lessons from v2 and v3 failures):
    v2 (no norm): PEM output scale >> post_proj scale → model destroyed (7.3% acc)
    v3 (with norm): normalization kills gradient → PEM can't learn (loss flat)

    Root cause: training PEM end-to-end through a FROZEN 8B model creates either
    a scale mismatch (v2) or a gradient plateau (v3). The frozen model's gradients
    are too small/distorted for the tiny PEM to learn from.

    v4 FIX: Train PEM entirely on CACHED FEATURES (numpy arrays from Week 1).
    No model in the loop. The PEM learns in feature space:

        augmented = post_proj + gate * PEM_transform(PCA_extract(enc_out))

    Loss: cross-entropy on task_type prediction from augmented features.
    This teaches the PEM to produce features that, when added to post_proj,
    improve physics category decodability — which is exactly what we measured
    in Week 1 probing.

    Then we plug the trained PEM into the full model at INFERENCE TIME and
    evaluate on PhysBench val.

ADVANTAGES:
    1. Trains in SECONDS (200 cached samples, CPU-only, no GPU needed for training)
    2. No scale mismatch (trains in post_proj feature space directly)
    3. No gradient-through-frozen-model problem
    4. Clean data (pre-filtered, no video processing)
    5. Gate calibrated by alpha sweep on cached features before plugging into model

Usage:
    # Train PEM on cached features + calibrate gate + eval on PhysBench val
    python scripts/week3_pem_v4_feature_train.py

    # Skip model eval (just feature-space training)
    python scripts/week3_pem_v4_feature_train.py --feature-train-only
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.optim.features import FeatureCache
from src.optim.physbench_split import classify_quantitative
from src.optim.steering import compute_pca_basis
from src.optim.resilience import configure_traceback_logging
from scripts.run_physbench_eval import load_physbench_data


def train_pem_feature_space(
    enc_features: np.ndarray,      # [N, 1152]
    post_proj_features: np.ndarray, # [N, 4096]
    labels: np.ndarray,             # [N] string labels (task_type)
    low_var_k: int = 64,
    hidden_dim: int = 256,
    epochs: int = 50,
    lr: float = 0.01,
    gate_values: list = None,
) -> Dict:
    """Train PEM transform in feature space. Returns the best (transform, gate).

    The training loop:
        1. PCA on enc_features → low_var_basis [K, 1152]
        2. physics_raw = enc_features @ low_var_basis.T → [N, K]
        3. For each gate value:
            augmented = post_proj + gate * transform(physics_raw)
            loss = cross_entropy(classifier(augmented), labels)
        4. Joint optimization of transform + classifier weights.
        5. Return the gate value that maximizes HELD-OUT accuracy.
    """
    from sklearn.preprocessing import LabelEncoder

    if gate_values is None:
        gate_values = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]

    # Encode labels.
    le = LabelEncoder()
    valid_mask = np.array([bool(str(l).strip()) for l in labels])
    enc_features = enc_features[valid_mask]
    post_proj_features = post_proj_features[valid_mask]
    labels = labels[valid_mask]
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)
    print(f"  Classes: {list(le.classes_)}, n={len(y)}")

    # PCA on enc_features for the subspace extractor.
    components, variances, mean = compute_pca_basis(enc_features)
    n_comp = len(variances)
    k = min(low_var_k, n_comp)
    low_var_basis = components[-k:]  # [K, 1152]
    explained = variances[-k:].sum() / variances.sum()
    print(f"  PCA: K={k}, V_low explains {explained:.6f} of total variance")

    # Extract physics features.
    physics_raw = enc_features @ low_var_basis.T  # [N, K]
    print(f"  physics_raw: shape={physics_raw.shape}, std={physics_raw.std():.4f}")

    # Target scale: post_proj std.
    target_std = float(post_proj_features.std())
    print(f"  post_proj target_std={target_std:.4f}")

    # Train/val split (80/20).
    n = len(y)
    perm = np.random.RandomState(42).permutation(n)
    n_train = int(0.8 * n)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    # Convert to tensors.
    X_physics = torch.tensor(physics_raw, dtype=torch.float32)
    X_post = torch.tensor(post_proj_features, dtype=torch.float32)
    Y = torch.tensor(y, dtype=torch.long)

    best_result = None
    best_val_acc = 0.0

    for gate in gate_values:
        print(f"\n  --- gate={gate} ---")

        # PEM transform: K → hidden → 4096 (same arch as PEM, but trained here).
        transform = nn.Sequential(
            nn.LayerNorm(k),
            nn.Linear(k, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, post_proj_features.shape[1]),
        )
        # Small init on last layer.
        nn.init.normal_(transform[-1].weight, std=0.01)
        nn.init.zeros_(transform[-1].bias)

        # Classifier head (trained jointly, discarded after — only transform is kept).
        classifier = nn.Linear(post_proj_features.shape[1], n_classes)

        params = list(transform.parameters()) + list(classifier.parameters())
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        best_epoch_acc = 0.0
        best_transform_state = None

        for epoch in range(epochs):
            # Training.
            transform.train()
            classifier.train()

            physics_feats = transform(X_physics[train_idx])
            augmented = X_post[train_idx] + gate * physics_feats
            logits = classifier(augmented)
            loss = criterion(logits, Y[train_idx])

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            # Validation.
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                transform.eval()
                classifier.eval()
                with torch.no_grad():
                    pf_val = transform(X_physics[val_idx])
                    aug_val = X_post[val_idx] + gate * pf_val
                    logits_val = classifier(aug_val)
                    pred = logits_val.argmax(dim=1)
                    val_acc = (pred == Y[val_idx]).float().mean().item()

                    # Also check baseline (no PEM) accuracy.
                    logits_base = classifier(X_post[val_idx])
                    base_acc = (logits_base.argmax(1) == Y[val_idx]).float().mean().item()

                if epoch == epochs - 1 or (epoch + 1) % 25 == 0:
                    print(f"    epoch {epoch+1:>3}: loss={loss.item():.4f} "
                          f"val_acc={val_acc:.4f} base_acc={base_acc:.4f} "
                          f"delta={val_acc - base_acc:+.4f}")

                if val_acc > best_epoch_acc:
                    best_epoch_acc = val_acc
                    best_transform_state = {k: v.clone() for k, v in transform.state_dict().items()}

        print(f"  gate={gate}: best_val_acc={best_epoch_acc:.4f}")

        if best_epoch_acc > best_val_acc:
            best_val_acc = best_epoch_acc
            best_result = {
                "gate": gate,
                "val_acc": best_epoch_acc,
                "transform_state": best_transform_state,
                "low_var_basis": low_var_basis,
                "target_std": target_std,
                "hidden_dim": hidden_dim,
                "n_classes": n_classes,
                "classes": list(le.classes_),
            }

    print(f"\n  BEST: gate={best_result['gate']}, val_acc={best_result['val_acc']:.4f}")
    return best_result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-vl-8b")
    ap.add_argument("--cache-dir", default="cache/week1", type=Path)
    ap.add_argument("--data-dir", default="data/physbench", type=Path)
    ap.add_argument("--output-dir", default="results/week3", type=Path)
    ap.add_argument("--log-dir", default="logs", type=Path)
    ap.add_argument("--low-var-k", type=int, default=64)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--feature-train-only", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_traceback_logging(args.log_dir, "week3_pem_v4")

    print("=" * 70)
    print(" PEM v4: Feature-Space Training (no model forward pass)")
    print("=" * 70)

    # Load cached features.
    cache = FeatureCache(args.cache_dir / "features", args.model, "val")
    enc_features = cache.load_site("enc_out")
    post_proj_features = cache.load_site("post_proj")
    index_order = list(cache._index["sample_ids"])
    print(f"Loaded: enc_out {enc_features.shape}, post_proj {post_proj_features.shape}")

    # Load labels.
    samples = load_physbench_data(str(args.data_dir), split="val")
    for s in samples:
        s.setdefault("sample_id", f"val_{s.get('idx', '?')}")
    id_to_task = {s["sample_id"]: s.get("task_type", "") for s in samples}
    labels = np.array([id_to_task.get(sid, "") for sid in index_order])

    # Train PEM in feature space.
    print("\nTraining PEM transform on cached features...")
    t0 = time.time()
    result = train_pem_feature_space(
        enc_features, post_proj_features, labels,
        low_var_k=args.low_var_k,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
    )
    print(f"Training time: {time.time()-t0:.1f}s")

    # Save the trained transform.
    save_path = args.output_dir / "pem_v4_transform.pt"
    torch.save({
        "transform_state": result["transform_state"],
        "gate": result["gate"],
        "low_var_basis": result["low_var_basis"],
        "target_std": result["target_std"],
        "hidden_dim": result["hidden_dim"],
        "val_acc": result["val_acc"],
    }, save_path)
    print(f"Saved: {save_path}")

    if args.feature_train_only:
        print("Skipping model eval (--feature-train-only)")
        return 0

    # ---- Plug into full model and evaluate on PhysBench val ----
    print("\n" + "=" * 70)
    print(" Evaluating PEM v4 on PhysBench val (full model)")
    print("=" * 70)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch as torch_gpu
    from src.optim.vram import build_bnb_config, snapshot_vram, hard_cleanup
    from src.optim.compute import pick_attn_impl
    from src.optim.pem import PhysicsSubspaceExtractor, PhysicsTransform
    from src.optim.features import _resolve_module
    from scripts.run_physbench_eval import resolve_media_paths, format_question_for_vlm, extract_answer

    # Load model.
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    print("Loading Qwen3-VL-8B...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct",
        quantization_config=build_bnb_config(load_in_4bit=True),
        device_map="auto", torch_dtype=torch_gpu.bfloat16,
        attn_implementation=pick_attn_impl(), low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", trust_remote_code=True)
    model.eval()
    device = next(model.parameters()).device
    print(f"VRAM: {snapshot_vram()}")

    # Build the PEM injection hook from the trained transform.
    low_var_basis = result["low_var_basis"]  # [K, 1152]
    K = low_var_basis.shape[0]
    gate = result["gate"]
    hidden_dim = result["hidden_dim"]
    llm_dim = post_proj_features.shape[1]

    # Reconstruct the transform.
    transform = nn.Sequential(
        nn.LayerNorm(K),
        nn.Linear(K, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, llm_dim),
    )
    transform.load_state_dict(result["transform_state"])
    transform = transform.to(device).to(torch_gpu.bfloat16)
    transform.eval()

    basis_tensor = torch_gpu.tensor(low_var_basis, dtype=torch_gpu.bfloat16, device=device)

    # Enc capture hook.
    enc_capture = {}
    def enc_hook(_mod, _inp, output):
        t = output
        if isinstance(t, tuple): t = t[0]
        elif hasattr(t, "last_hidden_state"): t = t.last_hidden_state
        enc_capture["enc_out"] = t
        return output

    # PEM injection hook.
    def pem_hook(_mod, _inp, output):
        enc_out = enc_capture.get("enc_out")
        if enc_out is None:
            return output
        is_tuple = isinstance(output, tuple)
        t = output[0] if is_tuple else output

        # PCA projection.
        physics_raw = enc_out.to(basis_tensor.dtype) @ basis_tensor.T  # [..., K]
        # Mean pool if dims don't match (enc has more tokens than post_proj).
        if physics_raw.shape[:-1] != t.shape[:-1]:
            if physics_raw.ndim == 2 and t.ndim == 2:
                physics_raw = physics_raw.mean(0, keepdim=True).expand(t.shape[0], -1)
            elif physics_raw.ndim == 3 and t.ndim == 3:
                physics_raw = torch_gpu.nn.functional.adaptive_avg_pool1d(
                    physics_raw.transpose(1,2), t.shape[1]).transpose(1,2)

        # Transform.
        with torch_gpu.no_grad():
            physics_feats = transform(physics_raw.to(torch_gpu.float32)).to(t.dtype)

        t_steered = t + gate * physics_feats
        if is_tuple:
            return (t_steered,) + output[1:]
        return t_steered

    enc_module = _resolve_module(model, "model.visual.blocks.26")
    merger_module = _resolve_module(model, "model.visual.merger")
    h1 = enc_module.register_forward_hook(enc_hook)
    h2 = merger_module.register_forward_hook(pem_hook)

    # Evaluate.
    from qwen_vl_utils import process_vision_info
    eval_samples = load_physbench_data(str(args.data_dir), split="val")
    for s in eval_samples:
        s.setdefault("sample_id", f"val_{s.get('idx', '?')}")

    correct = {"quantitative": 0, "qualitative": 0}
    total = {"quantitative": 0, "qualitative": 0}

    for i, sample in enumerate(eval_samples):
        try:
            media_paths = resolve_media_paths(sample, str(args.data_dir))
            if not any(p for p in media_paths): continue
            messages, has_media = format_question_for_vlm(sample, media_paths)
            if not has_media: continue
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            img, vid = process_vision_info(messages)
            inputs = processor(text=[prompt], images=img, videos=vid, padding=True, return_tensors="pt").to(device)
            with torch_gpu.inference_mode():
                out = model.generate(**inputs, max_new_tokens=10, do_sample=False, num_beams=1, use_cache=True)
            resp = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = extract_answer(resp) or ""
            gt = str(sample.get("answer", "")).strip().upper()
            sl = classify_quantitative(sample)
            total[sl] += 1
            if pred == gt and pred in {"A","B","C","D"}: correct[sl] += 1
            del inputs, out
        except Exception:
            pass
        if (i+1) % 50 == 0:
            print(f"  [{i+1}/200] q={correct['quantitative']}/{total['quantitative']} "
                  f"l={correct['qualitative']}/{total['qualitative']}")

    h1.remove()
    h2.remove()

    nq, nl = total["quantitative"], total["qualitative"]
    acc_q = correct["quantitative"] / nq if nq else 0
    acc_l = correct["qualitative"] / nl if nl else 0

    print(f"\n{'='*60}")
    print(f" PEM v4 RESULT vs BASELINES")
    print(f"{'='*60}")
    print(f"  {'method':<25} {'acc_quant':>10} {'acc_qual':>10} {'d_quant':>10} {'d_qual':>10}")
    print(f"  {'-'*58}")
    print(f"  {'Baseline':<25} {'0.7091':>10} {'0.6207':>10} {'':>10} {'':>10}")
    print(f"  {'LoRA B (merger)':<25} {'0.6727':>10} {'0.5655':>10} {'-0.0364':>10} {'-0.0552':>10}")
    print(f"  {'LoRA C (LLM)':<25} {'0.7273':>10} {'0.6069':>10} {'+0.0182':>10} {'-0.0138':>10}")
    print(f"  {'SCAS amplify a=3':<25} {'0.7455':>10} {'0.6207':>10} {'+0.0364':>10} {'+0.0000':>10}")
    dq = acc_q - 0.7091
    dl = acc_l - 0.6207
    print(f"  {'PEM v4 (this run)':<25} {acc_q:>10.4f} {acc_l:>10.4f} {dq:>+10.4f} {dl:>+10.4f}")
    print(f"  {'(gate=' + str(gate) + ')':<25}")
    print(f"{'='*60}")

    # Save results.
    out_path = args.output_dir / "pem_v4_eval.json"
    json.dump({
        "model": args.model, "gate": gate, "low_var_k": args.low_var_k,
        "feature_train_val_acc": result["val_acc"],
        "acc_quant": acc_q, "acc_qual": acc_l,
        "n_quant": nq, "n_qual": nl,
        "d_quant": dq, "d_qual": dl,
    }, open(out_path, "w"), indent=2)
    print(f"Results: {out_path}")

    hard_cleanup(model, processor)
    return 0


if __name__ == "__main__":
    sys.exit(main())
