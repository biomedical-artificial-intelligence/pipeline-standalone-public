# Antibody CDR / Peptide Design Pipeline (standalone)

A file-based command-line pipeline that designs **antibody CDR** sequences or
free **peptides** by iteratively folding candidates with **AlphaFold3**, docking
them against a target with **ZDOCK**, and steering the search with a **UCB
bandit** (same engine + early-stopping for both modes; pick via `--mode`).

This is a **standalone** build: it does **not** bundle AlphaFold3 and runs no
web server. You install AlphaFold3 yourself and point the pipeline at it via
environment variables, and you obtain the ZDOCK toolkit yourself (see
Requirements).
Everything runs from the command line and reads/writes plain files plus a small
SQLite job database.

> **AlphaFold3 requirement:** the pipeline calls AF3 as a normal subprocess —
> `"$FOLD_PYTHON" "$FOLD_SCRIPT" --json_path=… --run_inference=true`.
> Your AF3 must be runnable that way (a Python env where `run_alphafold.py`
> works). Running AF3 only via a Docker image is **not** supported out of the
> box — run this pipeline *inside* that image, or install AF3 into a venv.

```
bandit (propose CDR)  ->  AlphaFold3 (fold 2-chain antibody)  ->  ZDOCK (dock vs antigen)  ->  score  ->  bandit
```

---

## 1. Requirements

| Component | What | How the pipeline finds it |
|-----------|------|----------------------------|
| **AlphaFold3** | Your own AF3 install (code + model weights + databases) — **you provide this** | `ALPHAFOLD_*` env vars |
| **ZDOCK 3.0.2** | `zdock`, `mark_sur`, `create.pl`, `create_lig`, `block.pl`, `uniCHARMM` (+ the legacy `libg2c.so.0` that `mark_sur` links against) — **you provide this**, see step 2c | `DOCK_DIR` env var |
| **distance_antibody** | Lab Fortran helper — source ships in `dock_src/`, compile with gfortran | `DISTANCE_ANTIBODY_BIN` env var |

The pipeline's own Python dependencies (numpy, scipy, aiosqlite, PyMOL) are
installed via conda — see below.

**Host tools** (standard on most Linux; install if missing): `conda`/`mamba`
(Python env), `gfortran` (builds the distance helper), and `perl` at
`/usr/bin/perl` (ZDOCK's `create.pl`). The ZDOCK 3.0.2 Linux build is x86-64
and dynamically linked against glibc/libstdc++ plus the legacy `libg2c.so.0`,
so install your distribution's g77 compatibility package if that library is
missing.

---

## 2. Install

```bash
# 2a. Python environment (pipeline deps only — NOT AlphaFold3)
conda env create -f environment.yml
conda activate pipeline-standalone

# 2b. Build the lab distance helpers (antibody + peptide mode)
gfortran -O2 -o dock_src/distance_antibody dock_src/distance_antibody.f   # antibody
gfortran -O2 -o dock_src/distance          dock_src/distance.f            # peptide

# 2c. Obtain ZDOCK 3.0.2 yourself from https://zlab.umassmed.edu/zdock/
#     (free for academic/non-profit use; it is NOT redistributed here).
#     Unpack it anywhere and point DOCK_DIR at that directory with an
#     ABSOLUTE path — the pipeline runs from app/, so a relative path breaks.
#     mark_sur also needs the legacy libg2c.so.0 next to the binaries; on most
#     distributions it comes from the compat package for g77 (e.g. libg2c0).
#     Keep uniCHARMM in DOCK_DIR as well: mark_sur reads it from its cwd,
#     and the pipeline runs mark_sur with cwd=DOCK_DIR.
# 2d. Install AlphaFold3 separately (its own environment / Docker image).

# 2e. Fetch the example target structure (chain A of PDB 3WZE, VEGFR-2 kinase
#     domain). Structures are not redistributed here; this reproduces the exact
#     file the examples expect:
{ curl -sL https://files.rcsb.org/download/3WZE.pdb | grep '^ATOM'; echo END; } \
  > examples/3WZE.pdb
```

---

## 3. Configure

Copy the template and edit it for your machine (use absolute paths). A `.env`
at the repository root is **loaded automatically**; `.env.example` documents
every variable, and the defaults live in `app/config.py`.

```bash
cp .env.example .env
$EDITOR .env        # fill in the paths
```

`.env` is git-ignored, so your paths stay local.

The variables you must set:

```bash
# External AlphaFold3 — must be runnable as: $FOLD_PYTHON $FOLD_SCRIPT ...
FOLD_PYTHON=/path/to/af3-venv/bin/python
FOLD_SCRIPT=/path/to/alphafold3/run_alphafold.py
FOLD_MODEL_DIR=/path/to/alphafold3/models
FOLD_DB_DIR=/path/to/alphafold3/databases

# ZDOCK toolkit you unpacked in step 2c — ABSOLUTE path
DOCK_DIR=/abs/path/to/zdock3.0.2

# Lab Fortran helper built in step 2b — ABSOLUTE path
DISTANCE_ANTIBODY_BIN=/abs/path/to/repo/dock_src/distance_antibody
```

(`mark_sur` needs the legacy `libg2c.so.0`. Put it in `DOCK_DIR` next to the
binaries — the pipeline prepends `DOCK_DIR` to `LD_LIBRARY_PATH` automatically,
so nothing else is needed.)

---

## 4. Run

The quickest start is the known-good example (real 3WZE parameters). It needs
`examples/3WZE.pdb` from install step 2e:

```bash
bash examples/run_3WZE.sh          # smoke test: max_iter 5, warmup 2
# fuller run:  MAX_ITER=2000 WARMUP_SIZE=51 bash examples/run_3WZE.sh
```

Use it as a template for your own target. The underlying command is:

```bash
cd app
python run_pipeline.py \
  --pdb_name /path/to/antigen.pdb \
  --target_residue A:100 \
  --fixed_chain_sequence EVQLVESGGGLVQPGG... \
  --initial_cdr ARDYYGSSYW \
  --prefix    WGQGTLVTVSS... \
  --suffix    ...FGGGTKVEIK \
  --selected_chain_type Heavy \
  --gpu_ids 0,1,2 \
  --max_iter 2000
```

> Tip: keep each long sequence on a single line — a stray space/line-break
> inside `--fixed_chain_sequence` etc. silently splits it into a bad argument.
> Running via a script (like `examples/run_3WZE.sh`) avoids that.

### Peptide mode (`--mode peptide`)

Designs a free single-chain peptide against a target — same UCB bandit +
early-stopping engine, just a 1-chain AlphaFold3 fold and the peptide distance
helper (`dock_src/distance`, `DISTANCE_MAX = 15 Å`).

```bash
bash examples/run_peptide_3WZE.sh        # smoke test

# underlying command:
cd app
python run_pipeline.py \
  --mode peptide \
  --pdb_name /path/to/target.pdb \
  --target_residue A:100 \
  --initial_cdr ARDYYGSSYW \          # seed PEPTIDE (length = design length)
  --gpu_ids 0,1,2 --max_iter 2000
```

Antibody-only args (`--fixed_chain_sequence`, `--prefix`, `--suffix`,
`--selected_chain_type`) are unused in peptide mode. Requires `DISTANCE_BIN` in
`.env` and the `dock_src/distance` binary built (see `dock_src/README.md`).

### Arguments

| Arg | Meaning |
|-----|---------|
| `--pdb_name` | Receptor (antigen) PDB file |
| `--target_residue` | Target residue on the antigen, `CHAIN:RESNUM` (e.g. `A:100`) |
| `--fixed_chain_sequence` | The antibody chain that is **not** being designed |
| `--initial_cdr` | Seed CDR sequence (its length defines the CDR length) |
| `--prefix` / `--suffix` | Framework sequence flanking the CDR on the designed chain |
| `--selected_chain_type` | `Light` or `Heavy` — which chain carries the designed CDR |
| `--gpu_ids` | Comma-separated GPU ids, one AlphaFold worker per GPU (e.g. `0,1,2`) |
| `--max_iter` | Maximum search iterations |
| `--alpha` | Recorded with the run only — does **not** affect the search (see §6) |
| `--dock_workers_per_gpu` | ZDOCK parallelism per GPU (default `10`) |
| `--warmup_size` | Number of random warm-up sequences (default `3 × CDR length`); set small (e.g. `2`) for a quick smoke test |
| `--warmup_steps` | Search steps excluded from early-stop detection — the initial random-exploration region (default `10 × CDR length`); see §6 |
| `--early_stop` / `--no_early_stop` | Toggle early stopping on score plateau (default on); see §6 |
| `--es_window_size` / `--es_slope_eps` / `--es_patience_windows` | Early-stop tuning; see §6 |
| `--distance_processes` | Worker processes for the distance/pose-ranking pool (default `2`) |

> The designed chain sequence given to AlphaFold3 is `prefix + CDR + suffix`.
> ZDOCK parallelism is `len(gpu_ids) * dock_workers_per_gpu`.

---

## 5. Output

Results land in `app/output/<pipeline_id>/` (override with `OUTPUT_DIR`):

- `pipeline.log` — run log
- `warmup.json` — seed + warm-up sequences
- `mutation_history.json` — per-iteration sequence, score, and parent lineage
- `<iter>/` — per-iteration fold + docking artifacts (complex PDB, ZDOCK out)
- SQLite DB (`pipeline.db` by default) — pipeline + per-iteration scores

The best CDR sequence and score are printed at the end and stored in the DB.

---

## 6. Tuning the search & parallelism

All the knobs below are command-line flags — no source edits needed.

### Warm-up phase
Before the bandit starts steering, the pipeline folds + docks a batch of random
CDR sequences to seed its statistics.

- **Size** — `--warmup_size N` (default `3 × CDR length`). Each warm-up sequence
  is a full fold+dock, so this dominates start-up time. Use `--warmup_size 2`
  for a smoke test; keep the default (or larger) for a real search.
- After warm-up the bandit emits `n_dock_workers × 2` initial search jobs from
  the best warm-up sequence.

### Exploration → UCB transition
The search phase first explores **randomly**, then switches to UCB-guided
1-site mutation once enough observations exist.

- The transition threshold is **internal** (`random_phase_steps = 10 × CDR
  length`) and is **not** a command-line flag. Proposals before it are random;
  after it the bandit picks 1-site mutations by UCB.
- **`--warmup_steps N`** (default `10 × CDR length`) controls a related knob:
  how many initial search steps are **excluded from early-stop detection**, so
  early stopping does not fire during the random-exploration region. Raise it if
  early stop triggers too soon; lower it to let early stop watch sooner.
- (Other search constants — top-K pool size `_TOP_K = 10`, the 80%-best /
  20%-pool base-sequence pick — remain in `app/services/bandit/service.py`.)

### Early stopping
On by default — stops when the best score plateaus.

- **`--early_stop`** (default on) / **`--no_early_stop`** to run the full
  `--max_iter`.
- **`--warmup_steps N`** (default `10 × CDR length`) — initial search steps
  excluded from early-stop detection, so it won't stop during random exploration.
- Sensitivity: **`--es_window_size`** (default 100), **`--es_slope_eps`**
  (default 0.01; larger = stop sooner), **`--es_patience_windows`**
  (default 5; smaller = stop sooner).

### Parallelism & CPU usage
- **GPU / AlphaFold**: one AF worker per id in `--gpu_ids` (folds run in
  parallel, one per GPU).
- **ZDOCK (CPU-bound)**: `n_dock_workers = len(gpu_ids) × --dock_workers_per_gpu`
  (default 10/GPU). This sizes the docking thread pool — it is the main CPU
  throttle. Lower `--dock_workers_per_gpu` if the machine is overloaded, raise
  it to use more cores.
- **Distance calc**: each pose-ranking step uses a `multiprocessing.Pool` sized
  by **`--distance_processes`** (default 2). Bump it for faster per-job
  filtering on many-core machines.

### `--alpha`
Currently **recorded with the run (DB) but does not change the search** — the
UCB exploration constant is internal. Leave it at the default.

---

## Notes

- **GPU indexing**: `--gpu_ids` refers to the GPU ids visible to *your*
  AlphaFold3 process (`CUDA_VISIBLE_DEVICES` is set per worker).
- **Chain labelling**: the designed (CDR) chain is folded under the label from
  `--selected_chain_type` (`L`/`H`) so the distance helper can locate the CDR;
  the fixed chain takes the other label. The same `cdr_chain_id` flows through
  fold → dock → distance, so the CDR is always measured on the right chain.
- **Not bundled**: the ZDOCK toolkit, AlphaFold3 code / model weights /
  databases, and the example target structure. You provide all of them — this
  repository ships only its own source code (`app/`, `dock_src/`, `examples/`
  scripts). See install steps 2c and 2e.

---

## License

This project is licensed under the Apache License 2.0 — see the
[LICENSE](LICENSE) file for details.

### Third-party components

The Apache-2.0 license above covers **this repository's own code only**. The
following required components are *not* distributed here and carry their own
terms:

| Component | Location | Terms |
|---|---|---|
| ZDOCK 3.0.2 | not included | Weng Lab, UMass Chan Medical School. **Free for academic and non-profit use only**; commercial licensing is handled separately by BIOVIA. It is not redistributed here — obtain it from <https://zlab.umassmed.edu/zdock/>. |
| `libg2c.so.0` | not included | GNU g77 runtime library, required by `mark_sur`. Install it from your distribution's g77 compatibility package. |
| AlphaFold 3 | not included | Google DeepMind. The **source code** is CC BY-NC-SA 4.0; the **model parameters** must be requested from Google and are governed by the [AlphaFold 3 Model Parameters Terms of Use](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md). Neither is redistributed here. Nothing in this repository is derived from AlphaFold 3 source code — the pipeline writes AF3 input JSON and invokes `run_alphafold.py` as a subprocess. |
| PDB structures (e.g. 3WZE) | not included | Fetched from the RCSB PDB at <https://www.rcsb.org/> — see install step 2e. |

### AlphaFold 3 output

The structures this pipeline predicts are AlphaFold 3 Output. They are governed
by the [AlphaFold 3 Output Terms of Use](https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md),
**not** by the Apache-2.0 license above. In particular:

- Output is for **non-commercial use only**. Under those Terms only
  non-commercial organizations — universities, non-profit organizations and
  research institutes, educational, journalism and government bodies — may use
  it, and Output must not be shared with a commercial organization.
- If you publish or distribute Output you must give **conspicuous notice** that
  it is provided under and subject to those Terms, and you must **cite the
  AlphaFold 3 paper** (see Citation below).
- Making Output publicly available through a scientific publication or an open
  source release is expressly permitted by those Terms.

Read the Terms yourself before relying on this summary; it is not legal advice
and the Terms govern.

## Citation

If you use this pipeline, please cite the tools it depends on:

- **AlphaFold 3** — Abramson, J. *et al.* Accurate structure prediction of
  biomolecular interactions with AlphaFold 3. *Nature* **630**, 493–500 (2024).
  <https://doi.org/10.1038/s41586-024-07487-w>
- **ZDOCK 3.0.2** — Pierce, B. G., Hourai, Y. & Weng, Z. Accelerating protein
  docking in ZDOCK using an advanced 3D convolution library. *PLoS ONE*
  **6**(9), e24657 (2011). <https://doi.org/10.1371/journal.pone.0024657>
