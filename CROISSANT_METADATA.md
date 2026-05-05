# Croissant ML metadata for PhysBench-Diag

NeurIPS 2026 E&D Track requires the Croissant machine-readable metadata format with both **core** and **Responsible AI (RAI)** fields for any released dataset. PhysBench-Diag is a *derivative resource* of PhysBench v2 (Chow et al., 2024) — we add a deterministic quant/qual partition and per-model probing JSONs.

## Hosting plan

| Artifact | Host | URL (post-anonymization) |
|---|---|---|
| Code repository | GitHub (anonymous fork) | `github.com/anonymous-NeurIPS2026/physbench-diag` |
| Dataset (labels + probing JSONs) | Hugging Face Datasets | `huggingface.co/datasets/anonymous-NeurIPS2026/PhysBench-Diag` |
| Croissant metadata | Same HF dataset (root `croissant.json`) | discoverable via HF schema |
| Underlying PhysBench v2 images/videos | NOT redistributed | Users obtain from PhysBench's official release |

## File: `paper/croissant_metadata.json` (template — fill before submission)

```json
{
  "@context": {
    "@vocab": "https://schema.org/",
    "ml": "https://mlcommons.org/croissant/",
    "rai": "https://mlcommons.org/croissant/RAI/"
  },
  "@type": "Dataset",
  "name": "PhysBench-Diag",
  "description": "A diagnostic benchmark distilled from PhysBench v2 with a deterministic quantitative/qualitative sub-task partition. Used in conjunction with per-model probing JSONs and permutation tests to evaluate where in vision-language model architectures physics-reasoning failures originate.",
  "citation": "@inproceedings{anonymous2026physbench_diag, title={Compression Mechanism, Not Ratio, Predicts Quantitative-Physics Failures in Vision-Language Models}, author={Anonymous}, booktitle={NeurIPS 2026 Evaluations and Datasets Track}, year={2026}}",
  "license": "https://creativecommons.org/licenses/by/4.0/",
  "url": "https://huggingface.co/datasets/anonymous-NeurIPS2026/PhysBench-Diag",
  "version": "1.0.0",
  "creator": {"@type": "Person", "name": "Anonymous (NeurIPS 2026 submission)"},
  "datePublished": "2026-05-06",
  "keywords": [
    "vision-language models",
    "physics reasoning",
    "probing",
    "evaluation methodology",
    "compression",
    "pre-registered study"
  ],

  "ml:isLiveDataset": false,

  "rai:dataCollection": "Inherited deterministically from PhysBench v2 (Chow et al., 2024). No new data collection.",
  "rai:dataAnnotationProtocol": "Annotations (quant/qual) are deterministic functions of PhysBench v2 sub_type labels via src/optim/physbench_split.py:classify_quantitative(). No human annotators.",
  "rai:dataAnnotationPlatform": "N/A (deterministic algorithm)",
  "rai:dataAnnotationAnalysis": "Inter-annotator agreement: N/A (deterministic). Validation: Section 4.1 of paper describes the operational quant/qual mapping.",
  "rai:dataPreprocessingProtocol": "PhysBench v2 items are split using PhysBench's official val (200) and test (999) splits. Per-model probing applies bnb-NF4 4-bit quantization (1 model bf16) and forward hooks at 4 architectural sites.",
  "rai:dataReleaseMaintenancePlan": "Hosted on Hugging Face Datasets; maintained by paper author via GitHub. Errata via repository ERRATA.md.",
  "rai:dataUseCases": [
    "Layer-wise probing of vision-language models on physics-reasoning tasks",
    "Cross-VLM comparison stratified by quantitative vs qualitative physics tasks",
    "Validation of compression-bottleneck hypotheses",
    "Reproduction and stress-testing of compression-vs-degradation predictors"
  ],
  "rai:dataLimitations": [
    "n=200 validation items is small for some statistical tests; use n=999 test split for higher power",
    "14 of 200 validation items have unresolvable media paths in our local PhysBench mirror; documented as no_media in probing JSONs",
    "Quant/qual partition is derived from PhysBench sub_type taxonomy; if PhysBench updates the taxonomy, partition needs re-derivation",
    "Probing-feature caches are model-specific; new VLMs require new probing runs"
  ],
  "rai:dataSocialImpact": "Positive: enables more targeted improvement of VLMs on quantitative-physics tasks. Negative: probing tools could theoretically be repurposed for adversarial analysis; we release only standard linear probes that do not enable novel attack capabilities beyond existing probing literature.",
  "rai:dataBiases": "Inherited from PhysBench v2; please refer to PhysBench's original analysis. Our partition does not introduce new biases.",
  "rai:personalSensitiveInformation": "None. PhysBench items are public physics-reasoning prompts.",

  "distribution": [
    {
      "@type": "FileObject",
      "name": "physbench_diag_partition.json",
      "description": "Deterministic quant/qual labels for PhysBench v2 val + test splits.",
      "contentUrl": "https://huggingface.co/datasets/anonymous-NeurIPS2026/PhysBench-Diag/resolve/main/physbench_diag_partition.json",
      "encodingFormat": "application/json",
      "sha256": "TBD"
    },
    {
      "@type": "FileObject",
      "name": "probing_results_n10.zip",
      "description": "Per-model probing JSONs (10 models × {answer, task_type, sub_type} targets × {enc_out, post_proj, llm_8, llm_16} sites × {all, quantitative, qualitative} slices), plus 6 permutation_check JSONs.",
      "contentUrl": "https://huggingface.co/datasets/anonymous-NeurIPS2026/PhysBench-Diag/resolve/main/probing_results_n10.zip",
      "encodingFormat": "application/zip",
      "sha256": "TBD"
    }
  ],

  "recordSet": [
    {
      "@type": "RecordSet",
      "@id": "physbench_diag_items",
      "name": "PhysBench-Diag items",
      "description": "Each record is one PhysBench v2 item with the added quant/qual label.",
      "field": [
        {"@type": "Field", "name": "sample_id", "dataType": "Text", "description": "Unique sample identifier (e.g., val_0, test_42)"},
        {"@type": "Field", "name": "split", "dataType": "Text", "description": "val or test"},
        {"@type": "Field", "name": "task_type", "dataType": "Text", "description": "PhysBench taxonomy: dynamics / properties / relationships / scenes"},
        {"@type": "Field", "name": "sub_type", "dataType": "Text", "description": "PhysBench fine-grained 19-way label"},
        {"@type": "Field", "name": "quant_qual", "dataType": "Text", "description": "Derived: quantitative or qualitative"},
        {"@type": "Field", "name": "answer", "dataType": "Text", "description": "Correct answer letter A/B/C/D"}
      ]
    }
  ]
}
```

## Validation steps before submission

```bash
# 1. Install Croissant validator
pip install mlcroissant

# 2. Validate the JSON-LD
python -c "from mlcroissant import Dataset; d = Dataset(jsonld='paper/croissant_metadata.json'); print('valid' if d.metadata else 'invalid')"

# 3. Compute file SHA-256s and substitute into the "sha256" fields
shasum -a 256 results/week1_turing/*.json > paper/sha256_inventory.txt

# 4. Upload to Hugging Face Datasets:
#    - Create dataset repo `anonymous-NeurIPS2026/PhysBench-Diag`
#    - Upload croissant_metadata.json as the dataset-card root file
#    - Upload zipped probing JSONs
#    - Verify https://huggingface.co/datasets/<id>/croissant.jsonld resolves
```

## Why HuggingFace Datasets (vs Kaggle / Dataverse / OpenML)

- HF Datasets has the most ML-developer-native API and is one of the 4 explicitly-blessed hosts in the E&D call
- Built-in Croissant validator and dataset-card auto-generation
- Free, anonymizable for double-blind review
- Direct integration with the `datasets` Python library (one-line load)
