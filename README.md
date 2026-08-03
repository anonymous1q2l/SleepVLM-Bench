# SleepVLM-Bench

This is a clean-room reconstruction of the code structure required by the current
SleepVLM-Bench manuscript. The legacy scripts remain untouched in `../dataset`.

The project is **not yet a reproduction of the paper's numerical tables**. It first
establishes an auditable pipeline so that every reported value can be traced to a
cohort, subject split, sample manifest, model revision, prompt, raw response, parser,
and metric implementation.

## Pipeline

```text
raw PSG + annotations
        -> standardized continuous-signal preprocessing
        -> aligned 30-second epochs
        -> epoch manifest with subject identity
        -> deterministic 70/10/20 subject split
        -> optional diagnostic class sampling
        -> aligned EEG/EOG/EMG images and single-EEG sequences
        -> zero-shot VLM / LoRA VLM / sequence baseline
        -> immutable JSONL predictions
        -> deterministic parsing and metrics
        -> paper tables generated from run reports
```

## Current status

| Component | Status |
|---|---|
| Manifest schema and validation | Implemented |
| Subject-disjoint split | Implemented |
| Deterministic class sampling | Implemented |
| Paper preprocessing primitives | Implemented; cohort-wide QC remains pending |
| Three-channel rendering | Implemented |
| Paper prompts and prompt hashing | Implemented |
| First/last/majority parsers | Implemented |
| Accuracy, macro-F1, macro-recall, true kappa | Implemented |
| Prediction and image provenance schema | Implemented |
| OpenAI-compatible API runner | Implemented |
| SHHS raw-data adapter | Implemented; one real recording smoke-tested |
| ISRUC raw-data adapter | Implemented; one real recording smoke-tested |
| DCSM raw-data adapter | Implemented but unvalidated; raw EDF is absent locally |
| Local Gemma/Qwen/Llama/LLaVA inference | Interface and configs only |
| LoRA fine-tuning | Interface and configs only; legacy evidence is absent |
| ChatTS/DeepSleep/AttnSleep | External baseline contracts only |
| Paper table reproduction | Blocked until original predictions are recovered or rerun |

See [`docs/paper_code_mapping.md`](docs/paper_code_mapping.md) and
[`docs/implementation_status.md`](docs/implementation_status.md) before running data jobs.
The module layout is documented in [`docs/project_structure.md`](docs/project_structure.md).

## Repository scope

This repository contains benchmark infrastructure, configuration templates, and tests.
It does not contain PSG recordings, derived clinical images, model weights, API keys, or
the manuscript's historical predictions. LoRA and baseline configuration files describe
the intended experiment contract; they do not claim that those experiments have run.

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev,api]'
```

No credential is stored in the repository. API runners read keys from environment
variables such as `OPENAI_API_KEY` or a provider-specific variable named in a run config.

## Data setup

The committed dataset configs use portable paths under `data/raw/`. Either create local
symlinks at those paths or copy a config into the ignored `configs/local/` directory and
edit its paths. Do not commit local paths or any raw/derived PSG data.

```bash
mkdir -p configs/local
cp configs/datasets/isruc.toml configs/local/isruc.toml
# Edit configs/local/isruc.toml so raw_root points to the licensed local dataset.
```

Dataset acquisition is intentionally not automated. Access, use, and redistribution
must follow each cohort's license and ethics requirements. See
[`docs/data_access_and_privacy.md`](docs/data_access_and_privacy.md).

## Core commands

```bash
# Validate an epoch manifest and report cohort/stage/split counts.
sleepvlm validate-manifest --manifest manifests/epochs.csv

# Prepare one cohort. Start with --limit 1 and inspect the failure report.
sleepvlm prepare-dataset --config configs/datasets/isruc.toml \
  --output-root artifacts/prepared --manifest manifests/isruc.csv --limit 1

# Assign deterministic subject-disjoint splits independently within each cohort.
sleepvlm split-manifest --manifest manifests/epochs.csv \
  --output manifests/epochs_split.csv --seed 2024

# Select exact diagnostic counts from one split.
sleepvlm sample-manifest --manifest manifests/epochs_split.csv \
  --output manifests/isruc_diagnostic.csv --split test --cohort ISRUC \
  --targets W=100,N1=100,N2=100,N3=100,REM=100 --seed 2024

# Render images and write their SHA-256 values into a new manifest.
sleepvlm render-manifest --manifest manifests/isruc_diagnostic.csv \
  --output-root artifacts/images \
  --output-manifest manifests/isruc_diagnostic_images.csv

# Parse raw model responses and calculate metrics. Invalid outputs remain errors.
sleepvlm evaluate --predictions outputs/run_id/predictions.jsonl \
  --output outputs/run_id/metrics.json --parser first

# Convert one or more metric reports into auditable table rows.
sleepvlm aggregate-reports --reports outputs/*/metrics.json \
  --output outputs/tables/zero_shot.csv
```

Use `sleepvlm --help` and `sleepvlm <command> --help` for complete arguments. API and
local-model runners are deliberately not shown as one-click paper commands because all
models remain disabled until an exact revision and smoke test have been recorded.

## Tests

The test suite uses synthetic data and does not require access to a clinical cohort:

```bash
pytest -q
```

Before a data run, start with `prepare-dataset --limit 1`, inspect the generated failure
report and waveform images, and only then increase the limit.

## Data and API safety

Waveform images and model prompts derived from PSG may still be sensitive research data.
Do not send DCSM or any restricted cohort to an external API without documented approval.
Keep `artifacts/`, `manifests/`, and `outputs/` out of public commits even when identifiers
appear pseudonymous.

Commands never infer a ground-truth label from a filename. Labels and subject IDs must
come from the manifest, which is the source of truth for all modalities and models.

## Reproducibility rules

1. Split subjects before sampling epochs.
2. Never tune prompts, parsers, or hyperparameters on the test split.
3. Keep invalid model outputs in the denominator.
4. Report macro-F1 as macro-F1; weighted-F1 is a separate optional metric.
5. Use standard Cohen's kappa as the main kappa metric. Any fixed-chance estimate must
   be labeled separately.
6. Preserve raw responses. Parsed labels alone are insufficient evidence.
7. A paper table row is publishable only when its run manifest and report exist.

## Contributing and release status

Development rules are in [`CONTRIBUTING.md`](CONTRIBUTING.md). The repository is an alpha
research reconstruction and has no selected open-source license yet. Choose a license,
confirm authorship/citation metadata, and complete
[`docs/release_checklist.md`](docs/release_checklist.md) before a public release.
