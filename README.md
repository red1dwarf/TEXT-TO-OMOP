# Text-to-OMOP (TEXT-TO-OMOP)

This repository contains the code and artifacts used for a **Text-to-SQL benchmark grounded in the OMOP Common Data Model (CDM v5.4)**, with an explicit **safe abstention** option (`<NO_SQL>`) for unanswerable requests.

It includes:
- dataset split utilities,
- LoRA fine-tuning (Llama-3 Instruct format),
- inference scripts (LoRA + schema-injected baselines),
- evaluation protocol (text + optional execution-based),
- post-hoc diagnostics,
- utilities to profile an OMOP PostgreSQL schema,
- and an optional PostgreSQL dump for a synthetic OMOP database.

---

## Repository structure

```
TEXT-TO-OMOP/
  DATA/
    unique_templates.jsonl
    Paraphrasing_generation_prompt.txt
    Unanswerable_dataset.jsonl
    UNANSWERABLES.md
    Answerable_dataset.json
    train.jsonl
    val.jsonl
    test.jsonl
    split_script.py
  FT/
    Fine-tuning.py
  INFERENCE/
    INF-LORA/
      inference_lora.py
      results_lora_checkpoint-150.jsonl
      results_lora_checkpoint-312.jsonl
    INF-BASELINES/
      inference_schema_baselines.py
      inference_omop_schema.sql
      results_llama3_base.jsonl
      results_mistral_7b_instruct_v0_3.jsonl
      results_natural_sql_7b.jsonl
  EVALUATION/
    evaluation_protocol.py
  ANALYSIS/
    post-hoc_analysis.py
  DB/
    omop_profile.py
    DUMP.sql
    DB_generation_prompt.txt
```

## Data format

### Train/val/test JSONL (`train.jsonl`, `val.jsonl`, `test.jsonl`)
Each line is a JSON object with:
```json
{"input": "<question>", "output": "<sql or <NO_SQL>>"}
```

#### Output conventions (contract)

For every example and every model prediction:

- **Answerable**: `output` is a single SQL query that **must end with `;`**.
- **Unanswerable / abstention**: `output` is exactly `<NO_SQL>` (no other tokens).
- The inference scripts **truncate generations at the first `;`** and ensure the output is terminated with a semicolon.
- Empty/whitespace generations are normalized to `<NO_SQL>`.

### Source dataset (`Answerable_dataset.json`)
The split script expects a JSON **array** where each example contains at least:
- `template`
- `sql`
- `question_concept`  (the question text used for training/inference)

### Unanswerable questions list (`UNANSWERABLES.md`)
A plain-text list of unanswerable questions (one per line). Lines starting with `#` are treated as comments and ignored.

This file was created by **extracting the unanswerable questions present in the public EHRSQL/EHRSQL-2024 data files** and consolidating them into a single list for convenience (see “Data provenance & attribution” below).
The file starts with CC-BY attribution comments (lines beginning with `#`), which are ignored by the scripts.

### Normalized unanswerable set (`Unanswerable_dataset.jsonl`)

This file is a **normalized/tagged** version of `UNANSWERABLES.md`:
- `UNANSWERABLES.md` contains the **pooled raw unanswerable questions** extracted from the public EHRSQL/EHRSQL-2024 repository (one question per line).
- `Unanswerable_dataset.jsonl` contains the **same unanswerable questions**, but normalized and wrapped with the tagging format used throughout this benchmark (e.g., `<PERSON_ID>...</PERSON_ID>`, `<CONDITION>...</CONDITION>`, etc.).

**Note:** the split script currently consumes `UNANSWERABLES.md` (plain-text list). `Unanswerable_dataset.jsonl` is provided for reproducibility/analysis and for downstream uses requiring structured tags.

---

## Setup

### 1) Create an environment
Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate     # Windows PowerShell
```

### 2) Install dependencies
Minimal packages used by the scripts:

```bash
pip install -U pip
pip install torch transformers peft accelerate tqdm numpy pandas psycopg2-binary
```

Depending on your GPU / CUDA stack, you may want the official PyTorch install command from pytorch.org.

---

## Reproducing the pipeline

### A) Build train/val/test splits

The split utility creates train/val/test JSONL files from the source dataset, grouping by `(template, sql)` “variations” and keeping paraphrases together, with optional unanswerable questions appended as `<NO_SQL>`.  
(See `DATA/split_script.py` for the exact split logic and assumptions.)

```bash
python DATA/split_script.py \
  --input DATA/Answerable_dataset.json \
  --unanswerables DATA/UNANSWERABLES.md \
  --output_dir DATA \
  --seed 42
```

Outputs:
- `DATA/train.jsonl`
- `DATA/val.jsonl`
- `DATA/test.jsonl`

If you already provide `train.jsonl`, `val.jsonl`, `test.jsonl`, you can skip this step.

> Note: the split script is designed for the paper’s curated inventory (it assumes a specific grouping of variations/paraphrases). If you apply it to a different dataset, you may need to adapt the checks accordingly.

---

### B) Fine-tune a LoRA adapter (Llama-3 Instruct)

The fine-tuning script trains a LoRA adapter on a Llama-3 Instruct base model using the official Llama-3 chat token format and masks the prompt portion so loss is computed only on the SQL tokens.

```bash
python FT/Fine-tuning.py \
  --train_jsonl DATA/train.jsonl \
  --valid_jsonl DATA/val.jsonl \
  --output_dir FT/lora_text_to_omop \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --bf16 \
  --gradient_checkpointing \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --save_steps 250 \
  --eval_steps 250
```

Key arguments:
- `--target_modules` defaults to `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- `--resume_from_checkpoint` allows resuming Hugging Face trainer checkpoints
- `--do_sample_preview` runs a small deterministic generation preview after training

---

### C) Inference (LoRA adapters)

Runs inference on a JSONL test set for one or more LoRA adapters and writes one `results_*.jsonl` file per adapter.

```bash
python INFERENCE/INF-LORA/inference_lora.py \
  --test_jsonl DATA/test.jsonl \
  --output_dir INFERENCE/INF-LORA \
  --lora_dirs FT/lora_text_to_omop/checkpoint-150 FT/lora_text_to_omop/checkpoint-312 \
  --batch_size 8 \
  --max_new_tokens 1024 \
  --stop_at_semicolon_eos
```

Notes:
- Predictions are truncated to the first `;` and always terminated with a semicolon.
- Empty generations are converted to `<NO_SQL>`.
- `--merge_lora` can be used to merge LoRA weights into the base model for generation (requires enough VRAM).

---

### D) Inference (schema-injected baselines)

Runs baseline inference by injecting PostgreSQL DDL into every prompt. Supported baselines include:
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
- a Natural-SQL-7B model (see `--natural_sql_model_name`)

```bash
python INFERENCE/INF-BASELINES/inference_schema_baselines.py \
  --test_jsonl DATA/test.jsonl \
  --schema_path INFERENCE/INF-BASELINES/inference_omop_schema.sql \
  --output_dir INFERENCE/INF-BASELINES \
  --run_models llama3,mistral,natural \
  --batch_size 8 \
  --max_new_tokens 1024 \
  --stop_at_semicolon_eos
```

Outputs:
- `results_llama3_base.jsonl`
- `results_mistral_7b_instruct_v0_3.jsonl`
- `results_natural_sql_7b.jsonl`

---

## Evaluation

### A) Main evaluation protocol (text + optional execution)

The evaluation script computes:
- text-based metrics on answerable questions (EM, normalized EM, ROUGE-L),
- answerability detection (F1_ans),
- and (optionally) execution accuracy / F1_exe by running SQL in PostgreSQL in read-only mode with a timeout.

```bash
python EVALUATION/evaluation_protocol.py \
  --results_jsonl \
    INFERENCE/INF-LORA/results_lora_checkpoint-150.jsonl \
    INFERENCE/INF-LORA/results_lora_checkpoint-312.jsonl \
    INFERENCE/INF-BASELINES/results_llama3_base.jsonl
```

**Execution-based evaluation (optional):**
```bash
python EVALUATION/evaluation_protocol.py \
  --results_jsonl INFERENCE/INF-LORA/results_lora_checkpoint-312.jsonl \
  --pg_host localhost \
  --pg_port 5432 \
  --pg_db OMOP \
  --pg_user postgres \
  --pg_password <YOUR_PASSWORD> \
  --pg_schema omop \
  --pg_timeout_ms 5000 \
  --rs_c 0 1 10
```

> If `psycopg2` is not installed, execution-based evaluation is skipped.

---

### B) Post-hoc diagnostics table (6 metrics)

This script builds a CSV table of error diagnostics for result files, including:
- false abstain,
- schema drift (non-OMOP FROM/JOIN targets),
- base-table mismatch,
- DATEDIFF usage,
- CTE usage,
- visit_detail miss rate,
- plus unanswerable recall.

```bash
python ANALYSIS/post-hoc_analysis.py \
  --inputs \
    INFERENCE/INF-LORA/results_lora_checkpoint-150.jsonl \
    INFERENCE/INF-LORA/results_lora_checkpoint-312.jsonl \
    INFERENCE/INF-BASELINES/results_llama3_base.jsonl \
  --out ANALYSIS/table_6_metrics_corrected.csv
```

---

## Database utilities

### A) Profile an OMOP schema (PostgreSQL)

Produces per-table stats and OMOP coverage summaries as CSV, plus a `SUMMARY.txt`.

```bash
python DB/omop_profile.py \
  --host localhost \
  --port 5432 \
  --db OMOP \
  --user postgres \
  --password <YOUR_PASSWORD> \
  --schema omop \
  --out DB/omop_profile_out
```

### B) Restore the synthetic OMOP database dump (optional)

`DB/DUMP.sql` is a PostgreSQL `pg_dump` **plain SQL** dump. The dump is exported without owner/privileges metadata (no `OWNER TO`, no `GRANT`). Restore with:

```bash
createdb OMOP
psql -d OMOP -f DB/DUMP.sql
```

If the dump targets a specific schema, you may want to set `search_path` accordingly when evaluating.

---

## Results format

All inference scripts write JSONL with:
```json
{"input": "...", "gold": "...", "pred": "..."}
```

This is the expected input format for `EVALUATION/evaluation_protocol.py` and `ANALYSIS/post-hoc_analysis.py`.

---

## Data provenance & attribution

### Third-party resources (CC BY 4.0)

Parts of the question inventory and the unanswerable-question pool used in this repository were derived from the following public resources:

- **EHRSQL** (Lee et al., 2022): https://github.com/glee4810/EHRSQL  
  **License:** CC BY 4.0 (Creative Commons Attribution 4.0 International).

- **EHRSQL 2024 Shared Task materials** (Lee et al., 2024): https://github.com/glee4810/ehrsql-2024  
  **License:** CC BY 4.0 (Creative Commons Attribution 4.0 International).

**CC BY 4.0 summary:** you must give appropriate credit, provide a link to the license, and indicate if changes were made.  
License link: https://creativecommons.org/licenses/by/4.0/

### What was adapted in this repository

- `UNANSWERABLES.md`: consolidated pool of unanswerable questions extracted from the above resources, with minor normalization/clean-up (see header in `UNANSWERABLES.md`).
- `Unanswerable_dataset.jsonl`: tagged/normalized version of the same pool, aligned with this benchmark’s format.

If you reuse the data, please cite the original resources (Lee et al., 2022; Lee et al., 2024) and comply with their licenses/terms. No endorsement by the original authors is implied.

---

## Acknowledgements

This repository uses:
- Hugging Face `transformers` + `Trainer`
- `peft` for LoRA training/inference
- PostgreSQL for execution-based evaluation
