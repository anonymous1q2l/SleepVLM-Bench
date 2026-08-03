# Data access and privacy

## Data are not distributed here

This repository does not redistribute DCSM, SHHS, ISRUC, or derived PSG samples. Dataset
configuration files contain only expected directory layouts and channel names. Users are
responsible for obtaining each cohort through its authorized source and complying with
its data-use agreement, citation requirements, and institutional approvals.

## Local storage

Raw PSG, annotations, prepared NPZ files, waveform images, manifests, predictions, and
checkpoints belong in ignored local directories. Subject identifiers should be replaced
with project-scoped pseudonyms before any artifact is shared outside the approved team.
Removing names from a filename does not by itself make physiological data unrestricted.

## External model APIs

An external VLM request transmits the waveform image and prompt to a third party. Before
using `run-openai-compatible`, document that the dataset agreement and ethics approval
permit this transfer, verify provider retention/training terms, and use an approved
account and region. When approval is absent, use a locally hosted model.

API keys must be supplied through environment variables. Never place a key in TOML,
source code, shell history committed to the repository, logs, or prediction files.

## Public release check

Before publishing a commit, inspect tracked files and repository history for EDF, XML,
XLSX, NPZ, NPY, waveform images, predictions, checkpoints, local paths, and credentials.
The `.gitignore` is a guardrail, not a substitute for that inspection.

