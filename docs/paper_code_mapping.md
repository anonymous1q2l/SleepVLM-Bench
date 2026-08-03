# Paper-to-code mapping

| Manuscript claim | Canonical project location | Evidence required before reporting |
|---|---|---|
| Three cohorts | `configs/datasets/*.toml` | Recording inventory and exclusion report |
| Unified filtering and 100 Hz resampling | `data/preprocess.py` | Resolved config and QC summary per cohort |
| 30-second aligned epochs | Epoch manifest | Subject, onset, label, artifact index |
| 70/10/20 subject split | `data/split.py` | Split manifest and zero subject overlap check |
| Diagnostic class subset | `data/sample.py` | Seed, source split, requested and realized counts |
| EEG/EOG/EMG stacked image | `data/render.py` | Image path and SHA-256 for every sample ID |
| Image-only and rule prompts | `prompts/*.txt` | Prompt hash stored with every prediction |
| Six zero-shot VLMs | One resolved run per condition | Exact provider/model ID, revision, decoding config |
| Three LoRA VLMs and three seeds | `training/` contracts | Logs, adapters, selected hyperparameters, predictions |
| ChatTS/DeepSleep/AttnSleep | `baselines/` contracts | Shared test sample IDs and raw predictions |
| First-label parser | `evaluation/parser.py` | Parser name/version in metrics report |
| ACC/MF1/Recall/Kappa | `evaluation/metrics.py` | Report generated from immutable prediction JSONL |

## Publication gate

A result row is ready for the paper only when all of the following identifiers exist:

- `run_id`
- `cohort`
- `split_manifest_sha256`
- `model_id` and optional immutable revision
- `prompt_sha256`
- decoding or training configuration
- raw prediction JSONL
- parser mode
- metric report JSON

Legacy summary CSV values do not satisfy this gate because they do not preserve the raw
responses or a reliable mapping to model, cohort, prompt, and sample manifest.
