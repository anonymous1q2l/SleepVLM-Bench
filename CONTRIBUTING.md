# Contributing

## Principles

- Preserve the manifest as the only source of sample identity and ground truth.
- Never derive labels from filenames.
- Never commit clinical data, generated PSG images, model weights, credentials, or raw
  model responses from restricted cohorts.
- Do not add a paper result without its resolved config, sample-manifest hash, raw
  predictions, parser version, and generated metric report.
- Keep legacy imports isolated in explicit conversion tools with source hashes.

## Development workflow

1. Create a focused branch or change set.
2. Add or update synthetic tests for behavioral changes.
3. Run `pytest -q`.
4. Run a one-record smoke test for changes to cohort preprocessing.
5. Inspect the failure report and at least one rendered image per affected cohort.
6. Document changes that alter a preprocessing or evaluation protocol.

Do not weaken manifest validation to accommodate an old artifact. Write a versioned
migration tool and preserve the original file instead.

