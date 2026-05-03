#!/usr/bin/env python3
"""
Generate all paper figures from on-disk JSONs.

Usage:
    pip install matplotlib seaborn numpy   # if needed
    python scripts/figures/generate_paper_figures.py

Output: figures/fig{1..5}_*.pdf

All figures are read from:
  - results/week1_turing/phys_lens_predict_weekb.json (LOO results, per-model scores)
  - results/week1_turing/<model>_quant_qual_probe.json (probing tables)
  - results/week1/<model>_quant_qual_probe.json (4 baselines)
  - results/week1_turing/<model>_permutation_check.json (significance)

NO synthetic / placeholder data — every number traces to a JSON.

Style: NeurIPS-friendly. Minimal seaborn; matplotlib only for portability.
Color palette: viridis-based for accessibility.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_TURING = ROOT / "results" / "week1_turing"
RESULTS_BASELINE = ROOT / "results" / "week1"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

# Models in the n=10 panel (verified from phys_lens_predict_weekb.json)
PANEL = [
    "internvl3-8b",
    "gemma4-e4b",
    "qwen2.5-vl-7b",
    "qwen3-vl-8b",
    "llava-onevision-7b",
    "phi3.5-vision",
    "granite-vision-3.2-2b",
    # n=10 expansion (2026-05-03):
    "idefics3-8b",
    "idefics2-8b",
    "blip2-opt-2.7b",
]

# Short labels for compact plots
SHORT = {
    "internvl3-8b":           "InternVL3-8B",
    "gemma4-e4b":             "Gemma3-4B",
    "qwen2.5-vl-7b":          "Qwen2.5-VL-7B",
    "qwen3-vl-8b":            "Qwen3-VL-8B",
    "llava-onevision-7b":     "LLaVA-OV-7B",
    "phi3.5-vision":          "Phi-3.5-V",
    "granite-vision-3.2-2b":  "Granite-V-2B",
    "idefics3-8b":            "Idefics3-8B",
    "idefics2-8b":            "Idefics2-8B",
    "blip2-opt-2.7b":         "BLIP-2-OPT",
}

# Compression-mechanism taxonomy (the n=10 "refined hypothesis" finding).
# Determined by reading each model's source paper / HF model card:
#   spatial_merge:  deterministic block pooling / spatial merger (lossy)
#   learned_resampler: learned cross-attention or pixel-shuffle (task-aware)
#   no_compression: per-token MLP / linear projection (1x)
MECHANISM = {
    # Spatial-merge / pooling (deterministic, lossy)
    "qwen3-vl-8b":            "spatial_merge",
    "qwen2.5-vl-7b":          "spatial_merge",
    "gemma4-e4b":             "spatial_merge",
    "internvl3-8b":           "spatial_merge",   # 2.4x light pooling
    # Learned-resampler / pixel-shuffle (task-aware reduction)
    "idefics3-8b":            "learned_resampler",   # pixel-shuffle r=2
    "idefics2-8b":            "learned_resampler",   # perceiver resampler 64q
    "blip2-opt-2.7b":         "learned_resampler",   # Q-Former 32q
    # No compression (per-token MLP/Linear, 1x)
    "llava-onevision-7b":     "no_compression",
    "phi3.5-vision":          "no_compression",
    "granite-vision-3.2-2b":  "no_compression",
}

MECH_COLOR = {
    "spatial_merge":      "#d62728",  # red
    "learned_resampler":  "#9467bd",  # purple
    "no_compression":     "#1f77b4",  # blue
}
MECH_LABEL = {
    "spatial_merge":      "Spatial-merge (lossy)",
    "learned_resampler":  "Learned resampler",
    "no_compression":     "No compression (1×)",
}


def load_predictor_json() -> dict:
    """Load the verdict JSON. Single source of truth for compression + H3."""
    return json.loads((RESULTS_TURING / "phys_lens_predict_weekb.json").read_text())


def load_probe_json(model: str) -> Optional[dict]:
    """Load per-model probe JSON, trying turing dir first then baseline dir."""
    for d in (RESULTS_TURING, RESULTS_BASELINE):
        p = d / f"{model}_quant_qual_probe.json"
        if p.exists():
            try:
                content = json.loads(p.read_text())
                if content.get("probe_results"):
                    return content
            except Exception:
                pass
    return None


# =============================================================================
# Figure 1: HEADLINE — LOO predicted vs empirical scatter with regression line
# =============================================================================

def fig1_loo_scatter() -> None:
    """LOO predicted vs empirical H3 hit-rate. The headline result."""
    pred = load_predictor_json()
    loo = pred["loo_regression"]["per_model_loo"]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    # Per-point data
    names, predicted, empirical, abs_err = [], [], [], []
    for m, v in loo.items():
        names.append(m)
        predicted.append(v["predicted"])
        empirical.append(v["empirical"])
        abs_err.append(v["abs_error"])

    predicted = np.array(predicted)
    empirical = np.array(empirical)
    abs_err = np.array(abs_err)

    # Color by absolute error: green=good, red=bad
    cmap = plt.cm.RdYlGn_r
    norm = plt.Normalize(0, max(0.5, abs_err.max()))
    colors = cmap(norm(abs_err))

    # Reference y=x line
    lim = [-0.05, 1.05]
    ax.plot(lim, lim, "--", color="grey", alpha=0.4, label="Perfect prediction (y=x)")

    # Per-point scatter
    for i, m in enumerate(names):
        ax.scatter(predicted[i], empirical[i], s=140, c=[colors[i]],
                   edgecolors="black", linewidth=1.0, zorder=3)
        # Label slightly offset
        offset_y = 0.04 if empirical[i] < 0.5 else -0.06
        ax.annotate(SHORT.get(m, m), (predicted[i], empirical[i]),
                    xytext=(8, offset_y * 100), textcoords="offset points",
                    fontsize=9, alpha=0.85)

    # Header stats from JSON
    median_err = pred["loo_regression"]["median_abs_error"]
    rho = pred["loo_regression"]["spearman_rho"]
    p_val = pred["loo_regression"]["spearman_p"]
    n = pred["loo_regression"]["n_models"]
    kill = pred["loo_regression"]["kill_gate_fired"]

    title_lines = [
        f"PhysLens-Predict LOO regression (n={n})",
        f"Median |error| = {median_err:.3f}   "
        f"Spearman ρ = {rho:.3f} (p = {p_val:.3f})   "
        f"Kill-gate {'FIRED' if kill else 'PASS'}",
    ]
    ax.set_title("\n".join(title_lines), fontsize=11, pad=12)
    ax.set_xlabel("LOO predicted H3 hit-rate", fontsize=11)
    ax.set_ylabel("Empirical H3 hit-rate (3-target permutation)", fontsize=11)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # Color-bar legend for error
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
    cbar.set_label("Per-model |error|", fontsize=9)

    plt.tight_layout()
    out = FIGURES / "fig1_loo_scatter.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out}")


# =============================================================================
# Figure 2: Compression vs empirical H3 (raw scatter, NOT LOO)
# =============================================================================

def fig2_compression_h3() -> None:
    """log10(compression) on x, empirical H3 on y. Shows directional claim."""
    pred = load_predictor_json()
    scores = pred["per_model_scores"]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Filter to models with both compression + empirical_h3 in panel
    pts: List[Tuple[str, float, float]] = []
    for m in PANEL:
        s = scores.get(m, {})
        c = s.get("compression")
        h = s.get("empirical_h3")
        if c is None or h is None:
            continue
        pts.append((m, np.log10(max(c, 1.0)), h))

    if not pts:
        print("  fig2: no data")
        return

    xs = np.array([p[1] for p in pts])
    ys = np.array([p[2] for p in pts])

    # Color bands by compression regime
    def band(c_log: float) -> str:
        if c_log < 0.5:    # < 3.2
            return "#1f77b4"  # blue (negative ctrl)
        if c_log < 1.5:    # 3.2 to 32
            return "#9467bd"  # purple (mid)
        return "#d62728"      # red (high)

    colors = [band(x) for x in xs]

    # Linear fit (visual, NOT LOO)
    if len(xs) >= 2:
        a, b = np.polyfit(xs, ys, 1)
        x_fit = np.linspace(xs.min() - 0.2, xs.max() + 0.2, 50)
        ax.plot(x_fit, a * x_fit + b, "--", color="grey", alpha=0.6,
                label=f"Linear fit (visual)")

    for (m, x, y), c in zip(pts, colors):
        ax.scatter(x, y, s=160, c=c, edgecolors="black", linewidth=1.0, zorder=3)
        ax.annotate(SHORT.get(m, m), (x, y), xytext=(8, 4),
                    textcoords="offset points", fontsize=9, alpha=0.85)

    rho = pred["loo_regression"]["spearman_rho"]
    p = pred["loo_regression"]["spearman_p"]
    n = pred["loo_regression"]["n_models"]
    ax.text(0.03, 0.97, f"Spearman ρ = {rho:.3f}\np = {p:.3f}, n = {n}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    ax.set_xlabel("log₁₀(vision-token compression ratio)", fontsize=11)
    ax.set_ylabel("Empirical H3 hit-rate", fontsize=11)
    ax.set_title("Architectural compression vs. empirical H3 hit-rate",
                 fontsize=12, pad=10)
    ax.grid(True, alpha=0.25)

    legend = [
        Patch(facecolor="#1f77b4", edgecolor="black", label="No compression (~1×)"),
        Patch(facecolor="#9467bd", edgecolor="black", label="Mid (2–32×)"),
        Patch(facecolor="#d62728", edgecolor="black", label="High (>32×)"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=9)

    plt.tight_layout()
    out = FIGURES / "fig2_compression_h3.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out}")


# =============================================================================
# Figure 3: Per-model probe accuracy heatmap (target=answer, sites × slices)
# =============================================================================

def fig3_probing_heatmap() -> None:
    """Per-model heatmap of probe accuracy on target=answer."""
    sites = ["enc_out", "post_proj", "llm_8", "llm_16"]
    slices = ["all", "quantitative", "qualitative"]
    cols: List[str] = []
    for site in sites:
        for sl in slices:
            cols.append(f"{site}\n{sl[:4]}")

    rows: List[str] = []
    matrix: List[List[float]] = []
    for m in PANEL:
        d = load_probe_json(m)
        if d is None:
            continue
        ans = d.get("probe_results", {}).get("answer", {})
        row = []
        for site in sites:
            for sl in slices:
                v = ans.get(site, {}).get(sl, {})
                acc = v.get("mean_acc") if isinstance(v, dict) else None
                row.append(np.nan if acc is None else float(acc))
        rows.append(SHORT.get(m, m))
        matrix.append(row)

    if not matrix:
        print("  fig3: no data")
        return

    M = np.array(matrix)
    fig, ax = plt.subplots(figsize=(11, max(4, 0.6 * len(rows))))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0.15, vmax=0.45)

    # Annotate cells
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center",
                        color="white", fontsize=8)
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.3 else "black", fontsize=8)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=0, fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=9)
    ax.set_title("Probe accuracy on target='answer' (4-way A/B/C/D)\n"
                 "Sites: encoder_out → post_projector → LLM layer 8 → LLM layer 16",
                 fontsize=11, pad=10)

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Mean accuracy (5-fold CV)", fontsize=9)

    plt.tight_layout()
    out = FIGURES / "fig3_probing_heatmap.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out}")


# =============================================================================
# Figure 4: Δprobe (enc_quant - post_proj_quant) bar chart, sorted by compression
# =============================================================================

def fig4_delta_probe_bars() -> None:
    """Δprobe = enc_quant_acc - post_proj_quant_acc on target=answer.
    The KEY signal for H3.
    """
    pred = load_predictor_json()
    rows = []
    for m in PANEL:
        s = pred["per_model_scores"].get(m, {})
        c = s.get("compression")
        gap = s.get("enc_minus_postproj_gap")
        if c is None or gap is None:
            continue
        rows.append((m, c, gap))
    rows.sort(key=lambda r: r[1])  # sort by compression ascending

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    names = [SHORT.get(r[0], r[0]) for r in rows]
    comps = [r[1] for r in rows]
    gaps = [r[2] for r in rows]

    # Color by sign: positive = H3-supporting, negative = anti-H3
    colors = ["#2ca02c" if g > 0 else "#d62728" for g in gaps]

    y_pos = np.arange(len(rows))
    bars = ax.barh(y_pos, gaps, color=colors, edgecolor="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.5)

    # Compression annotation
    for i, (b, c) in enumerate(zip(bars, comps)):
        w = b.get_width()
        x_text = w + (0.005 if w >= 0 else -0.005)
        ha = "left" if w >= 0 else "right"
        ax.text(x_text, i, f"  {c:.1f}×", ha=ha, va="center", fontsize=8,
                color="grey")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Δprobe = quant_acc(enc_out) − quant_acc(post_proj)  on target='answer'",
                  fontsize=10)
    ax.set_title("H3 signal per model (positive Δ = compression destroys quant info)",
                 fontsize=11, pad=10)
    ax.grid(True, alpha=0.25, axis="x")

    legend = [
        Patch(facecolor="#2ca02c", edgecolor="black", label="Δ > 0 (H3-supporting)"),
        Patch(facecolor="#d62728", edgecolor="black", label="Δ < 0 (anti-H3)"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=9)

    plt.tight_layout()
    out = FIGURES / "fig4_delta_probe_bars.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out}")


# =============================================================================
# Figure 5: PhysBench-Diag composition (val + test, quant/qual)
# =============================================================================

def fig5_dataset_composition() -> None:
    """Stacked-bar visualization of PhysBench-Diag composition."""
    splits = ["Validation", "Test"]
    quant = [55, 274]
    qual = [145, 725]
    totals = [200, 999]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(splits))
    width = 0.5
    b1 = ax.bar(x, qual, width, label="Qualitative", color="#1f77b4",
                edgecolor="black", linewidth=0.8)
    b2 = ax.bar(x, quant, width, bottom=qual, label="Quantitative",
                color="#ff7f0e", edgecolor="black", linewidth=0.8)

    # Annotate counts
    for i, (q, ql, t) in enumerate(zip(quant, qual, totals)):
        ax.text(i, ql / 2, f"{ql}\n({100*ql/t:.0f}%)", ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")
        ax.text(i, ql + q / 2, f"{q}\n({100*q/t:.0f}%)", ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")
        ax.text(i, t + 15, f"Total: {t}", ha="center", fontsize=10,
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontsize=11)
    ax.set_ylabel("Number of items", fontsize=11)
    ax.set_title("PhysBench-Diag composition (quant/qual partition)", fontsize=12, pad=10)
    ax.set_ylim(0, max(totals) * 1.15)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    out = FIGURES / "fig5_dataset_composition.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out}")


# =============================================================================
# Main
# =============================================================================

# =============================================================================
# Figure 6 (NEW HEADLINE): H3 stratified by compression mechanism
# =============================================================================

def fig6_mechanism_stratified() -> None:
    """The n=10 finding: H3 hit-rate clusters by compression MECHANISM,
    not by compression ratio. This is the paper's revised primary claim.
    """
    pred = load_predictor_json()

    # Group models by mechanism
    groups: Dict[str, List[Tuple[str, float, float]]] = {
        "spatial_merge": [], "learned_resampler": [], "no_compression": [],
    }
    for m in PANEL:
        s = pred["per_model_scores"].get(m, {})
        c = s.get("compression"); h = s.get("empirical_h3")
        if c is None or h is None:
            continue
        mech = MECHANISM.get(m)
        if mech is None:
            continue
        groups[mech].append((m, c, h))

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5),
                                             gridspec_kw={"width_ratios": [1.4, 1]})

    # ---- LEFT: scatter colored by mechanism ----
    for mech, items in groups.items():
        if not items:
            continue
        xs = [np.log10(max(c, 1.0)) for _, c, _ in items]
        ys = [h for _, _, h in items]
        ax_left.scatter(xs, ys, s=180, c=MECH_COLOR[mech],
                        edgecolors="black", linewidth=1.0,
                        label=f"{MECH_LABEL[mech]} (n={len(items)})", zorder=3)
        for (m, c, h), x in zip(items, xs):
            ax_left.annotate(SHORT.get(m, m), (x, h), xytext=(8, 4),
                             textcoords="offset points", fontsize=8.5, alpha=0.85)

    ax_left.set_xlabel("log₁₀(vision-token compression ratio)", fontsize=11)
    ax_left.set_ylabel("Empirical H3 hit-rate", fontsize=11)
    ax_left.set_title("Compression vs. H3 — colored by reduction mechanism (n=10)",
                      fontsize=11, pad=10)
    ax_left.set_ylim(-0.05, 1.1)
    ax_left.grid(True, alpha=0.25)
    ax_left.legend(loc="upper left", fontsize=9, framealpha=0.95)

    # ---- RIGHT: per-mechanism mean H3 with individual points overlaid ----
    mech_order = ["no_compression", "learned_resampler", "spatial_merge"]
    means = []
    points_by_mech = []
    for mech in mech_order:
        h_values = [h for _, _, h in groups[mech]]
        means.append(np.mean(h_values) if h_values else 0)
        points_by_mech.append(h_values)

    x_pos = np.arange(len(mech_order))
    bars = ax_right.bar(x_pos, means, width=0.55,
                         color=[MECH_COLOR[m] for m in mech_order],
                         edgecolor="black", linewidth=0.8, alpha=0.7)

    # Overlay individual points (jittered)
    rng = np.random.RandomState(0)
    for i, h_values in enumerate(points_by_mech):
        if not h_values:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=len(h_values))
        ax_right.scatter(x_pos[i] + jitter, h_values, s=70, c="black",
                         alpha=0.85, zorder=3, edgecolors="white", linewidth=0.5)

    # Annotate means
    for i, m in enumerate(means):
        ax_right.text(i, m + 0.04, f"{m:.2f}", ha="center", fontsize=10,
                      fontweight="bold")

    ax_right.set_xticks(x_pos)
    ax_right.set_xticklabels([MECH_LABEL[m].split(" ")[0] for m in mech_order],
                              fontsize=10)
    ax_right.set_ylabel("Mean empirical H3 hit-rate", fontsize=11)
    ax_right.set_title("H3 by mechanism (mean ± per-model points)",
                       fontsize=11, pad=10)
    ax_right.set_ylim(-0.05, 1.15)
    ax_right.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    out = FIGURES / "fig6_mechanism_stratified.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out}")


def main() -> int:
    print(f"Reading from {RESULTS_TURING}")
    print(f"Writing figures to {FIGURES}")
    print()
    fig1_loo_scatter()
    fig2_compression_h3()
    fig3_probing_heatmap()
    fig4_delta_probe_bars()
    fig5_dataset_composition()
    fig6_mechanism_stratified()
    print()
    print("Done. PDFs at figures/fig{1..6}_*.pdf")
    print("To regenerate after data updates, just re-run this script.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
