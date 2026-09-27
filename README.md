# Business Entity Resolution — Amazon ML Challenge 2026

Given business records from three noisy sources (names and addresses in the **US, India and
France**), find every Source 2 / Source 3 record that refers to the same business as each
Source 1 record. Scored with **macro F0.5** (precision counts twice as much as recall).

The solution is a classic entity-resolution pipeline, built for ~23 million records on a laptop:

```
raw TSVs ──► normalise ──► infer states ──► blocking ──► pair features ──► LightGBM ──► decision rule ──► TSV outputs
            (clean text,    (city → state   (TF-IDF top-k   (~50 similarity   (pair         (each S2/S3 record
             transliterate   learned from    per state)      & agreement       probability)  goes to its best S1,
             Indian scripts) S1)                             features)                        if confident enough)
```

> **Branches:** this is the **`v4`** branch — the v3 pipeline plus a city → state inference
> stage (`src/infer_state.py`). The v3 pipeline is on
> [`main`](https://github.com/aarnavpi8/amazon-ml/tree/main).

| Version | What changed | Validation F0.5 | Leaderboard |
|---|---|---|---|
| v1 | tuned rule on blocking similarities | 0.750 | – |
| v2 | LightGBM, 40 features | 0.9695 | 0.957 |
| v3 (`main`) | + house-number / street features, decision tuned at test decoy density | 0.9725* | – |
| **v4** (this branch) | + city → state inference (fixes 35% of French records having no state) | 0.9725*† | – |

† Validation covers only the US and India, where state inference changes almost nothing, so v4
matches v3 there. Its intended gain is on the French records, which only the leaderboard can measure.

\* at test-like decoy density (see [How the decision is tuned](#6-decision-rule-and-tuning)).

---

## Contents

1. [What you need](#what-you-need)
2. [Directory layout](#directory-layout)
3. [Setup](#setup)
4. [Put the data in place](#put-the-data-in-place)
5. [Run the whole pipeline](#run-the-whole-pipeline)
6. [Run it step by step](#run-it-step-by-step)
7. [Check and submit the output](#check-and-submit-the-output)
8. [Running on Google Colab](#running-on-google-colab)
9. [Configuration](#configuration)
10. [How the method works](#how-the-method-works)
11. [Diagnostics and extra tools](#diagnostics-and-extra-tools)
12. [Troubleshooting](#troubleshooting)
13. [Rules compliance](#rules-compliance)

---

## What you need

| | Requirement |
|---|---|
| Python | 3.10 or newer (developed and tested on **3.13**) |
| OS | Windows, Linux or macOS. Commands below use **bash** (on Windows: Git Bash). Every step is also a plain `python -m ...` command, so PowerShell works too. |
| RAM | **24 GB** is what it was tested on. Everything except training is processed one file / one country at a time and fits in ~8 GB; training peaks higher (see [Colab](#running-on-google-colab) for low-memory settings). |
| Disk | ~3 GB for the dataset + **~10 GB free** for intermediate files in `work/` + ~1 GB for outputs |
| GPU | **Not needed.** Everything runs on CPU (multi-threaded). |
| Time | ~70 minutes end-to-end on a 20-thread laptop CPU (table in [step by step](#run-it-step-by-step)) |
| Data | The challenge's `student_resource` bundle (not included in this repository) |

## Directory layout

The repository contains **only code and documentation**. The dataset and everything the pipeline
generates are git-ignored because they are large (and the dataset is not ours to redistribute).

```
amazon-ml/                               <- repository root
├── README.md                            <- this guide
├── .gitignore
├── eda.ipynb                            <- exploratory data analysis notebook (outputs cleared)
├── code/
│   └── business_entity_resolution/
│       ├── requirements.txt             <- pinned dependencies
│       ├── run_pipeline.sh              <- runs every stage in order
│       ├── src/                         <- all source code (one module per stage)
│       │   ├── config.py                <- paths (overridable by env vars) and CANDIDATE_K
│       │   ├── text_maps.py             <- legal forms, abbreviations, state/region tables
│       │   ├── translit.py              <- Indian scripts -> Latin (mined dictionary + rules)
│       │   ├── normalize.py             <- stage 1: clean names and addresses
│       │   ├── infer_state.py           <- stage 2: fill missing states from city names
│       │   ├── block.py                 <- stage 3: candidate generation (blocking)
│       │   ├── features.py              <- pair features used by the model
│       │   ├── train.py                 <- stage 4: train LightGBM
│       │   ├── tune.py                  <- stage 5: tune the decision rule
│       │   ├── predict.py               <- stage 6: score test, write outputs
│       │   ├── decision.py              <- "best S1 per record" decision rule
│       │   ├── submission.py            <- writes the two TSVs in the exact format
│       │   ├── metrics.py               <- macro F0.5 exactly as the challenge defines it
│       │   ├── eval_blocking.py         <- blocking recall report (diagnostic)
│       │   └── baseline.py              <- v1 rule-based baseline (diagnostic)
│       └── work/                        <- CREATED BY THE PIPELINE (git-ignored)
├── dataset/                             <- YOU ADD THIS (git-ignored), see below
└── output/                              <- CREATED BY THE PIPELINE (git-ignored)
    ├── matching_results.tsv             <- the file you upload to the leaderboard
    └── candidate_pairs.tsv              <- blocking candidates (needed for the final zip)
```

## Setup

```bash
# 1. Clone
git clone https://github.com/aarnavpi8/amazon-ml.git
cd amazon-ml

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/Scripts/activate        # Windows (Git Bash)
# .venv\Scripts\Activate.ps1         # Windows (PowerShell)
# source .venv/bin/activate          # Linux / macOS

# 3. Install the pinned dependencies
pip install -r code/business_entity_resolution/requirements.txt

# Optional, only for the EDA notebook:
pip install matplotlib ipykernel
```

All dependencies are MIT / BSD / Apache-2.0 licensed: polars, numpy, scipy, scikit-learn,
sparse-dot-topn, rapidfuzz, indic-transliteration, lightgbm.

## Put the data in place

Unzip the challenge's `student_resource` bundle into a folder called `dataset/` at the
repository root, so the TSVs end up **exactly** here:

```
amazon-ml/
└── dataset/
    └── student_resource/
        ├── dataset/
        │   ├── train/
        │   │   ├── train_source1.tsv
        │   │   ├── train_source2.tsv
        │   │   ├── train_source3.tsv
        │   │   └── train_ground_truth.tsv
        │   └── test/
        │       ├── test_source1.tsv
        │       ├── test_source2.tsv
        │       └── test_source3.tsv
        └── utils/
            └── validate_submission.py    <- the organisers' format checker
```

Quick check (should list 7 files):

```bash
ls dataset/student_resource/dataset/train dataset/student_resource/dataset/test
```

If your data lives somewhere else, don't move it — point the pipeline at it instead:

```bash
export ER_DATA_DIR=/path/to/folder/containing/train/and/test
```

(`ER_DATA_DIR` must be the folder that directly contains `train/` and `test/`.)

## Run the whole pipeline

With the virtual environment active, from the repository root:

```bash
bash code/business_entity_resolution/run_pipeline.sh
```

That runs all 7 stages and ends with `output/matching_results.tsv` and
`output/candidate_pairs.tsv`. Progress is printed for every stage.

If `python` on your PATH is not the virtual-environment one, pass it explicitly (use an
**absolute** path — the script changes directory):

```bash
PYTHON="$PWD/.venv/Scripts/python" bash code/business_entity_resolution/run_pipeline.sh   # Windows
PYTHON="$PWD/.venv/bin/python" bash code/business_entity_resolution/run_pipeline.sh       # Linux / macOS
```

## Run it step by step

Every stage is a module run from `code/business_entity_resolution/`. Each stage reads the
previous stage's files from `work/`, so you can stop after any stage and continue later.

```bash
cd code/business_entity_resolution
```

| # | Command | What it does | Writes | Time* |
|---|---|---|---|---|
| 1 | `python -m src.normalize` | Cleans names (legal forms, honorifics, d/b/a, domains, injected IDs), transliterates Indian scripts, parses addresses (abbreviations, states, house numbers). Mines the Indic → Latin dictionary from training pairs first. | `work/translit_dict.json`, `work/norm/*.parquet` | ~3.5 min |
| 2 | `python -m src.infer_state` | Learns a city → state table from Source 1 and fills in missing states (mainly French addresses that end with only a city). Updates `work/norm/` in place. | `work/norm/*.parquet` | ~2 min |
| 3 | `python -m src.block --split train --force` | Candidate generation for the training split (used to train and evaluate the model). | `work/cand/train/<country>.parquet` | ~17 min |
| 4 | `python -m src.block --split test --force` | Candidate generation for the test split. | `work/cand/test/<country>.parquet` | ~12 min |
| 5 | `python -m src.train --rebuild` | Builds pair features for a 20% sample of training entities, trains LightGBM (1500 rounds), saves validation scores. | `work/feat_train.parquet`, `work/model/lgbm.txt`, `work/model/scored_valid.parquet`, `work/model/decision.json` | ~22 min |
| 6 | `python -m src.tune` | Tunes the decision threshold and margin at the test set's decoy density. | updates `work/model/decision.json` | ~3 min |
| 7 | `python -m src.predict` | Builds features for all test candidates, scores them, applies the decision rule, writes both TSVs. | `output/*.tsv`, `work/model/scored_test/*.parquet` | ~10 min |

\* measured on a 20-thread laptop CPU with 24 GB RAM.

Useful flags:

- `--force` (block): recompute countries that already have a file. Without it, blocking **resumes**
  and skips finished countries — handy after an interruption, but you **must** use `--force`
  after changing normalisation, otherwise stale candidates are reused.
- `--rebuild` (train): recompute the cached feature table `work/feat_train.parquet`. Needed after
  changing features, normalisation or blocking.
- `--reuse-scores` (predict): skip feature building and scoring, only re-apply the decision rule to
  the saved test scores. Use it after re-running `src.tune`: a new submission in ~1 minute.
- `--limit N` (normalize): process only the first N rows of every file — a quick smoke test of the
  whole code path (write to a separate work dir: `--work-dir /tmp/er_smoke`).

## Check and submit the output

Always run the organisers' validator before uploading (from the `student_resource` folder):

```bash
cd dataset/student_resource
python utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir dataset/test --check-ids
```

It must print `PASS`. Then upload `output/matching_results.tsv` to the portal.

Output format (tab-separated, one row per test Source 1 entity, empty list = no match):

```
source1_entity_id	matched_entity_ids
S1-714132312	S2-637340732,S3-625880872,S3-867809779
S1-106407869	S3-585937637
S1-000000001
```

`candidate_pairs.tsv` has the same shape (`candidate_entity_ids` column) and lists every
candidate the model scored. Matches are always a subset of candidates.

Keep a copy of every uploaded file (the guidelines ask for version history), e.g.
`output/history/v3_house_features/matching_results.tsv`.

## Running on Google Colab

```python
# Cell 1 — code and dependencies
!git clone https://github.com/aarnavpi8/amazon-ml.git
%cd amazon-ml
!pip install -q -r code/business_entity_resolution/requirements.txt

# Cell 2 — data: upload student_resource to Google Drive first, then mount it
from google.colab import drive
drive.mount('/content/drive')
import os
os.environ["ER_DATA_DIR"] = "/content/drive/MyDrive/student_resource/dataset"   # contains train/ and test/
os.environ["ER_WORK_DIR"] = "/content/work"       # local disk: much faster than Drive
os.environ["ER_OUTPUT_DIR"] = "/content/drive/MyDrive/er_output"               # survives disconnects

# Cell 3 — run
!bash code/business_entity_resolution/run_pipeline.sh
```

Colab notes:

- **Run it as a script (above), not as notebook cells** — long cells are what get killed.
- **Memory:** standard Colab has ~12 GB RAM. Stages 1–4, 6 and 7 fit. Training (stage 5) uses a
  20% entity sample; if it runs out of memory, lower `SAMPLE_PCT` in `src/train.py` from 20 to 10
  (validation stays at 5%). Colab High-RAM runtimes need no change.
- **Disconnects:** blocking resumes per country if you rerun `src.block` *without* `--force`.
  Put `ER_WORK_DIR` on Drive if you expect disconnects (slower, but survives them).
- A GPU runtime is not needed.

## Configuration

Paths (environment variables, read in `src/config.py`):

| Variable | Default | Meaning |
|---|---|---|
| `ER_DATA_DIR` | `<repo>/dataset/student_resource/dataset` | folder with `train/` and `test/` |
| `ER_WORK_DIR` | `<repo>/code/business_entity_resolution/work` | intermediate files |
| `ER_OUTPUT_DIR` | `<repo>/output` | the two submission TSVs |

Main knobs:

| Setting | Where | Default | Effect |
|---|---|---|---|
| `CANDIDATE_K` | `src/config.py` | 5 | S1 candidates kept per S2/S3 record. 5 → 97.9% blocking recall, 8 → 98.1% but ~1.6× more pairs to score |
| `RETRIEVE_TOPN`, `PASS_KEEP` | `src/block.py` | 20, 6 | per-pass retrieval depth / kept after re-ranking |
| `MAX_DF` | `src/block.py` | 0.02 | ignore character 3-grams in >2% of records (4× faster blocking) |
| `SAMPLE_PCT`, `VALID_PCT` | `src/train.py` | 20, 5 | share of training entities used / held out for validation |
| `PARAMS` | `src/train.py` | lr 0.1, 127 leaves | LightGBM hyper-parameters |

## How the method works

### 1. Normalisation (`normalize.py`, `translit.py`, `text_maps.py`)

The EDA (`eda.ipynb`) showed the noise the generator injects; each step targets one pattern:

- **Names:** lower-case, strip accents, remove junk prefixes (`--`, `<<`), injected IDs
  (`(ID: 20566)`, 6+ digit numbers), `M/s`, country tags (`(India)`), split website names
  (`srgold.com` → `srgold`), collapse dotted abbreviations (`S.A.R.L.` → `sarl`), unify spellings
  (`private` → `pvt`, `centre` → `center`), mark `d/b/a`. Legal forms are split off into
  `name_legal`, leaving `name_core` (e.g. `ram marketing`).
- **Indian scripts:** ~18% of Indian matches pair an English name with a Devanagari / Tamil /
  Bengali / … one. A dictionary of 1.3k Indic words → English is **mined from the training
  ground truth** (aligning word by word), with rule-based transliteration as fallback.
  Cross-script name similarity went from a median of 11 to 100.
- **Addresses:** placeholders (`NULL`, `N/A`) removed, abbreviations expanded per country
  (`Rd` → road; in France `R.` → rue, `St` → saint), state names canonicalised (TX ↔ Texas,
  MH ↔ Maharashtra ↔ महाराष्ट्र, Nord → Hauts-de-France), house number and all numbers extracted.

### 2. State inference (`infer_state.py`)

35% of French S2/S3 addresses end with just a city. A **city → state table is learned from
Source 1** (which always has `…, city, state`), and missing states are filled in from it.
French records without a state: 34.9% → 3.1%.

### 3. Blocking (`block.py`)

Each S2/S3 record ("query") retrieves likely S1 records **within its own country**:

- **name pass** — TF-IDF on character 3-grams of `name_core`, top 20 by cosine;
- **address pass** — the same on the address (state removed): catches the ~5% of matches whose
  name was replaced by an invented brand name;
- **exact pass** — identical no-space name, or identical `house-number street-word` key.

Searches run **inside state blocks** (fast and precise); records mentioning several states search
each, records with no state search the whole country. Telangana and Andhra Pradesh share a block
(S2/S3 still label Hyderabad with the pre-2014 state). Each pass re-ranks by
`name_cos + addr_cos` and keeps 6; the union is capped at `CANDIDATE_K` per query.
On the full training set: **97.9% pair recall, 23 candidates per S1, best reachable F0.5 0.993**.

### 4. Pair features (`features.py`) — ~50 per pair, none country-specific

- blocking cosines and which pass found the pair;
- context: rank and score gap of the pair within its query's candidates and within its S1's;
- rapidfuzz similarities of names and addresses (ratio, partial, token-set, token-sort,
  Jaro-Winkler; name with look-alike digits fixed, `8aba` → `baba`);
- agreement flags: legal form, state, house number (equal / conflicting / one is the other with
  a digit dropped / edit distance / relative gap), shared numbers, street without numbers,
  acronym, script, d/b/a.

The house-number features target how the generator makes **decoys** (same business name at a
nearby number: `7265` vs `7269`) versus how it adds **noise** to true matches (dropped digits:
`118` → `11`).

### 5. Model (`train.py`)

LightGBM binary classifier (MIT licence, far below the 8B-parameter limit). Training rows are the
blocking candidates of a 20% hash-sample of training S1 entities (≈32M pairs, 14.5% positive);
5% of entities are held out for validation. Country is never a feature, so France (absent from
training) is scored by the same model.

### 6. Decision rule and tuning (`decision.py`, `tune.py`)

Every S2/S3 record belongs to at most one S1 (verified on the ground truth). So each record is
assigned to its **highest-probability S1** only if the probability ≥ `t` and it beats the
runner-up by ≥ `margin`. `t` and `margin` are grid-searched for macro F0.5 on the validation
entities. The test set has **~1.9× more decoy records per S1** than train, so decoy records in
the validation data are replicated to test density before tuning (otherwise the rule is too
lenient). Values for v3: `t = 0.725`, `margin = 0.6` (each run writes its own to
`work/model/decision.json`).

## Diagnostics and extra tools

```bash
cd code/business_entity_resolution
python -m src.eval_blocking     # blocking recall by K, by segment, sample of missed pairs (after stage 3)
python -m src.baseline          # v1 rule baseline — NOTE: overwrites output/*.tsv
```

`eda.ipynb` (repository root) is the exploratory analysis. Its outputs are cleared in the
repository; run all cells to regenerate them (~1.5 min, needs `matplotlib` and `ipykernel`).

## Troubleshooting

| Problem | Fix |
|---|---|
| `UnicodeEncodeError: 'charmap' codec` on Windows | the console can't print table borders: `set PYTHONIOENCODING=utf-8` (cmd), `$env:PYTHONIOENCODING="utf-8"` (PowerShell), or use Git Bash |
| `FileNotFoundError ... train_source1.tsv` | data not where expected — check the tree in [Put the data in place](#put-the-data-in-place) or set `ER_DATA_DIR` |
| `FileNotFoundError ... work/norm/...` | run the stages in order; each needs the previous stage's files |
| Results didn't change after editing normalisation | rerun blocking with `--force` and training with `--rebuild` (both cache their outputs) |
| `MemoryError` / process killed during training | lower `SAMPLE_PCT` in `src/train.py` |
| Validator says a matched ID is not in the candidates | only possible if `output/` mixes files from different runs — rerun `src.predict` |
| Leaderboard rejects the file | run the validator; the file must be tab-separated UTF-8 with the exact header |

## Rules compliance

- **No external data or services.** Everything is learned from the provided files: the
  transliteration dictionary and the city → state table come from the training / Source 1
  records. The small static tables in `text_maps.py` are general conventions (street-type
  abbreviations, legal-form spellings, state codes such as TX = Texas, and the regions of the 4
  French départements that appear in the data) — no geocoding, registry or API lookups.
- **Country is an open set.** Countries without specific rules fall back to the default ones,
  and every test Source 1 entity always gets a row.
- **Model licence and size:** LightGBM (MIT), a few MB.
