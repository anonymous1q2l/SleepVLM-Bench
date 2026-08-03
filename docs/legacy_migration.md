# Legacy migration map

The legacy directory `../dataset` remains read-only reference material. No result is
accepted merely because a legacy filename resembles a model or cohort name.

| Legacy material | New owner | Migration decision |
|---|---|---|
| `生成图片代码/DCSM.py` | `data/cohorts.py`, `data/preprocess.py` | Reimplemented; W-only sampling and filter mismatch removed |
| `生成图片代码/isruc.py` | `data/cohorts.py` | Reimplemented for all subjects; text labels advance every epoch |
| `生成图片代码/shhs.py` | `data/cohorts.py` | Reimplemented; stage 9/unknown excluded instead of mapped to W |
| `大模型/拼接图片.py` | `data/render.py` | Replaced with one aligned shared-axis render |
| `大模型/生成jsonl文件.py` | `prompts/`, `inference/runner.py` | Prompt text versioned and hashed |
| `调用api_dataset/*.py` | `inference/openai_compatible.py` | Credentials moved to environment variables; model ID is explicit |
| `大模型/统计结果.py` | `evaluation/` | Last-label and weighted-F1 mistakes removed |
| `all_json_results.csv` | None | Audit reference only; raw responses and identity are absent |
| `processed_results.json` | None | Audit reference only; provenance conflicts with the paper |

## Recovery rule

Historical raw responses may be imported only through a dedicated converter that
provides a verified sample ID, cohort, model ID, prompt condition, and source file hash.
Unverifiable summaries must not be copied into `outputs/` or regenerated paper tables.

