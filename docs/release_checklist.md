# Release checklist

## Required before uploading a private development snapshot

- [ ] `pytest -q` passes in a clean environment.
- [ ] Package installation and `sleepvlm --help` work.
- [ ] No raw/derived PSG, model weights, predictions, or credentials are tracked.
- [ ] Committed configs contain portable paths only.
- [ ] README status matches implemented behavior.
- [ ] Unverified models and experiments remain disabled.

## Required before a public open-source release

- [ ] Select and add an open-source license with owner approval.
- [ ] Add verified authors, affiliations, contact, and repository URL.
- [ ] Add `CITATION.cff` after paper title/authors/identifier are final.
- [ ] Confirm dataset names, links, licenses, and required acknowledgements.
- [ ] Confirm external API use is allowed for every evaluated cohort.
- [ ] Pin a tested environment or publish a lock file/container digest.
- [ ] Replace mutable model aliases with exact revisions where possible.
- [ ] Run secret and large-file scans over the full Git history.
- [ ] Verify all local Markdown links and release commands.

## Required before releasing paper results

- [ ] DCSM, SHHS, and ISRUC preprocessing QC is signed off.
- [ ] Subject splits have zero overlap and their manifest hashes are archived.
- [ ] Every model/condition/seed has raw predictions and a generated report.
- [ ] LoRA selection uses validation data only.
- [ ] VLM and baseline test sample IDs are identical where the paper claims comparability.
- [ ] Paper tables are generated from reports rather than manually copied.

