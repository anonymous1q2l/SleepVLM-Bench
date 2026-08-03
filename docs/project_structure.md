# Project structure

```text
SleepVLM-Bench/
├── configs/
│   ├── datasets/       # Cohort paths, channels, filters, paper sample counts
│   └── experiments/    # Zero-shot, LoRA, and sequence-baseline claims
├── prompts/            # Versioned prompt text; every run stores its SHA-256
├── src/sleepvlm_bench/
│   ├── data/           # Discovery, annotation parsing, preprocessing, split, render
│   ├── inference/      # API/local VLM backends and immutable prediction runner
│   ├── evaluation/     # Parser, metrics, cohort reports, table aggregation
│   ├── training/       # LoRA publication-gate contracts
│   ├── baselines/      # Shared-test-set contracts for sequence models
│   ├── schema.py       # Canonical epoch record
│   ├── provenance.py   # Hashing and atomic JSON/JSONL I/O
│   └── cli.py          # `sleepvlm` command-line entry point
├── tests/              # Synthetic regression tests; no clinical data required
├── manifests/          # Generated sample manifests, ignored except placeholder
├── outputs/            # Generated run artifacts, ignored except placeholder
└── artifacts/          # Prepared arrays/images; always generated, never committed
```

## Ownership boundaries

`data/` owns the ground truth and sample identity. Inference code may read labels for
reporting provenance, but it must never derive them from filenames or include them in a
prompt. Evaluation code consumes immutable raw responses and does not call a model.

LoRA and baseline packages are deliberately separate from zero-shot inference because
they have training-state, checkpoint, seed, and validation-selection requirements.

## Canonical artifacts

An epoch manifest is the join key across image, sequence, and baseline experiments.
Prepared recording NPZ files contain three aligned channels and source epoch indices.
Image filenames are cosmetic and are never parsed for labels.

Each inference run owns a directory containing:

```text
outputs/<run_id>/
├── run.json             # Model, prompt, manifest hash, decoding settings
├── predictions.jsonl    # Raw response or explicit error for every sample
├── parsed.jsonl         # Parser output; generated after inference
└── metrics.json         # Overall and per-cohort metrics
```

