# Implementation status and unresolved decisions

## Implemented foundation

The current reconstruction implements the contracts that prevent the known legacy
errors: filename-derived labels, subject leakage, nondeterministic sampling, last-label
parsing, skipped invalid responses, weighted-F1 labeled as macro-F1, and missing run
provenance.

## Validation still required

1. Confirm the exact channel names and line frequency for every recording, not only one
   example from each cohort.
2. Confirm whether the manuscript's diagnostic counts refer to all splits or only the
   held-out test split. The paper currently uses both descriptions.
3. Confirm DCSM raw-data location. Only processed DCSM artifacts were found locally.
4. Decide whether the 48 Hz anti-alias filter is applied to all modalities or only when
   downsampling from a source rate above 100 Hz.
5. Recover exact model revisions and raw predictions for every existing table row.
6. Resolve whether the historical `step3` results were incorrectly labeled as Gemma.
7. Recompute every metric from raw predictions. Do not import numbers from the legacy CSV.

## Modules intentionally not presented as complete

Local VLM inference, LoRA training, ChatTS, DeepSleepNet, and AttnSleep require
model-specific integration and hardware validation. Their configuration contracts are
present, but no paper result should be attributed to them until their adapters and tests
are implemented and a complete run artifact is produced.

