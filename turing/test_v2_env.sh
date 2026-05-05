#!/bin/bash
# =============================================================================
# GPU VERIFICATION for vla_physics_v2
# =============================================================================
# Submitted automatically by setup_v2_env.sh, or run manually:
#     sbatch turing/test_v2_env.sh
#
# Tests (in order, fail-fast):
#   T1: torch CUDA available + correct version + correct env path
#   T2: GPU matmul performance (sanity for silent CPU fallback)
#   T3: bitsandbytes 4-bit Linear layer on GPU
#   T4: transformers AutoTokenizer + AutoConfig load (CPU)
#   T5: Small model load with 4-bit quant (gpt2 — fast smoke test)
#   T6: HuggingFace Hub access (HF_TOKEN check, only if set)
#   T7: Project script imports + key paths exist
#
# All tests pass => write 'PASS' marker to ~/.vla_setup_state/gpu_test.done
# Any test fails  => exits with the failed test's number (T1=1, T2=2, ...)
# =============================================================================

#SBATCH -J v2-test
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH -t 0:30:00
#SBATCH --account=${SLURM_ACCOUNT:-default}
#SBATCH --export=ALL
#SBATCH --gres=gpu:A100:1
#SBATCH -D ${HOME}/VLAs
#SBATCH -o jobs/%x.%j.out

set -uo pipefail
mkdir -p jobs

readonly ENV_NAME="vla_physics_v2"
readonly STATE_DIR="${HOME}/.vla_setup_state"
mkdir -p "${STATE_DIR}"

echo "================================================================"
echo "v2 ENV GPU VERIFICATION"
echo "  date:     $(date)"
echo "  job:      ${SLURM_JOB_ID}"
echo "  node:     $(hostname)"
echo "  GPU:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "  driver:   $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
echo "================================================================"

# Activate env. Disable `set -u` around conda.sh — older conda versions
# reference unset vars in their activation logic.
set +u
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "/opt/conda/etc/profile.d/conda.sh"
fi

if ! conda activate "${ENV_NAME}" 2>/dev/null; then
    if ! source activate "${ENV_NAME}" 2>/dev/null; then
        set -u
        echo "FATAL: cannot activate ${ENV_NAME}"
        exit 99
    fi
fi
set -u

PY_PATH=$(command -v python)
echo "python: ${PY_PATH}"
if [[ "${PY_PATH}" != *"${ENV_NAME}"* ]]; then
    echo "FATAL: python is ${PY_PATH} — not in ${ENV_NAME}"
    exit 99
fi
echo ""

# ----- Run tests in single python session (fail-fast on first error) -----
python -u - <<'PY'
import sys, os, time, traceback

# Each test is a function returning (bool, message)
# Test order matters — later tests assume earlier ones passed

def t1_torch_cuda():
    """torch CUDA available, correct version, correct env"""
    import torch
    expected_version = "2.4.1+cu124"
    if torch.__version__ != expected_version:
        return False, f"torch version: got {torch.__version__}, expected {expected_version}"
    if "vla_physics_v2" not in torch.__file__:
        return False, f"torch loaded from outside env: {torch.__file__}"
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() = False"
    return True, f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}"

def t2_gpu_matmul():
    """GPU matmul speed sanity (catches silent CPU fallback)"""
    import torch
    x = torch.randn(2000, 2000, device='cuda')
    torch.cuda.synchronize()
    # Warmup
    for _ in range(2):
        _ = x @ x.T
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        _ = x @ x.T
    torch.cuda.synchronize()
    dt = (time.time() - t0) * 1000 / 5
    if dt > 200:
        return False, f"matmul too slow: {dt:.1f}ms (expected <100ms — likely CPU fallback)"
    return True, f"matmul {dt:.1f}ms (good)"

def t3_bnb_4bit():
    """bitsandbytes 4-bit linear layer works on GPU"""
    import torch
    import bitsandbytes as bnb
    import bitsandbytes.nn as bnb_nn
    lin = bnb_nn.Linear4bit(128, 64, bias=False).cuda()
    x = torch.randn(4, 128, device='cuda', dtype=torch.float16)
    y = lin(x)
    if y.shape != (4, 64):
        return False, f"bad output shape {tuple(y.shape)}"
    return True, f"bnb {bnb.__version__} 4-bit Linear OK, output {tuple(y.shape)}"

def t4_transformers_imports():
    """Core transformers classes importable"""
    from transformers import (
        AutoConfig, AutoTokenizer, AutoProcessor,
        AutoModelForCausalLM, AutoModel,
        BitsAndBytesConfig,
    )
    import transformers
    return True, f"transformers {transformers.__version__}"

def t5_4bit_model_load():
    """Load small model with 4-bit quant — exercises bnb+transformers integration.

    Uses facebook/opt-125m: real architecture, small (~500MB), public (no auth),
    canonical bnb 4-bit smoke test. Layers are large enough for bnb's NF4
    block-wise quantization (block size 64).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_id = "facebook/opt-125m"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        # device_map={"": 0} = explicit "put everything on cuda:0".
        # More reliable than "auto" on single-GPU nodes for bnb 4-bit models —
        # "auto" sometimes triggers layer-balancing logic that hangs/OOMs.
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map={"": 0},
        )
    except Exception as e:
        return False, f"4-bit model load failed: {type(e).__name__}: {e}"

    # Quick generation test
    inputs = tok("Hello", return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=3, do_sample=False)
    # Sanity: model is actually quantized (parameters are 4-bit)
    n_4bit = sum(1 for p in model.parameters() if hasattr(p, 'quant_state'))
    return True, f"4-bit opt-125m loaded ({n_4bit} 4-bit params), generated {tuple(out.shape)}"

def t6_hf_hub_access():
    """HuggingFace Hub access (only checks if HF_TOKEN set)"""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return True, "skipped (HF_TOKEN not set — set it to access gated models)"
    from huggingface_hub import whoami
    try:
        info = whoami(token=token)
        return True, f"logged in as {info['name']}"
    except Exception as e:
        return False, f"HF token invalid or hub unreachable: {e}"

def t7_project_imports():
    """Project's own scripts import without GPU complaints"""
    sys.path.insert(0, "${HOME}/VLAs")
    failed = []
    for modpath in [
        "scripts.week1_quant_qual_probe",
        "scripts.discover_probe_sites",
        "scripts.extract_training_features",
        "scripts.phys_lens_predict",
    ]:
        try:
            __import__(modpath)
        except Exception as e:
            failed.append(f"{modpath}: {type(e).__name__}")
    if failed:
        return False, "imports failed: " + "; ".join(failed)
    return True, "all 4 scripts import OK"

TESTS = [
    ("T1 torch CUDA available",      t1_torch_cuda),
    ("T2 GPU matmul speed",          t2_gpu_matmul),
    ("T3 bnb 4-bit Linear",          t3_bnb_4bit),
    ("T4 transformers imports",      t4_transformers_imports),
    ("T5 4-bit gpt2 model load",     t5_4bit_model_load),
    ("T6 HuggingFace Hub access",    t6_hf_hub_access),
    ("T7 project script imports",    t7_project_imports),
]

passed = 0
failed_at = None
for i, (name, fn) in enumerate(TESTS, 1):
    print(f"[{i}/{len(TESTS)}] {name}...", flush=True)
    try:
        ok, msg = fn()
    except Exception as e:
        ok, msg = False, f"EXCEPTION: {type(e).__name__}: {e}\n{traceback.format_exc()}"
    if ok:
        print(f"        PASS — {msg}")
        passed += 1
    else:
        print(f"        FAIL — {msg}")
        failed_at = i
        break

print()
print("=" * 64)
if failed_at is None:
    print(f"=== ALL GPU TESTS PASSED ({passed}/{len(TESTS)}) ===")
    sys.exit(0)
else:
    print(f"=== TEST {failed_at} FAILED ({passed}/{len(TESTS)} passed before failure) ===")
    sys.exit(failed_at)
PY

TEST_RC=$?
echo ""
echo "================================================================"
if [ ${TEST_RC} -eq 0 ]; then
    echo "GPU VERIFICATION: PASS"
    touch "${STATE_DIR}/gpu_test.done"
    echo "  marker written: ${STATE_DIR}/gpu_test.done"
    echo ""
    echo "Next steps:"
    echo "  1. bash turing/migrate_to_v2.sh        # update SLURM scripts"
    echo "  2. bash turing/submit_weekb_parallel.sh"
    exit 0
else
    echo "GPU VERIFICATION: FAIL (test ${TEST_RC} failed)"
    rm -f "${STATE_DIR}/gpu_test.done"
    echo ""
    echo "Re-run setup if you want to retry installs:"
    echo "  bash turing/setup_v2_env.sh"
    echo ""
    echo "Or fix manually then re-submit this test:"
    echo "  sbatch turing/test_v2_env.sh"
    exit ${TEST_RC}
fi
