# Antibody CDR Design Pipeline (standalone)

A file-based command-line pipeline that designs antibody CDR sequences by
iteratively folding candidate antibodies with **AlphaFold3**, docking them
against a target antigen with **ZDOCK**, and steering the search with a
**UCB bandit**.

This is a **standalone** build: it does **not** bundle AlphaFold3 and runs no
web server. You install AlphaFold3 yourself and point the pipeline at it via
environment variables (the ZDOCK toolkit *is* bundled, in `dock_bin/`).
Everything runs from the command line and reads/writes plain files plus a small
SQLite job database.

> **AlphaFold3 requirement:** the pipeline calls AF3 as a normal subprocess —
> `"$ALPHAFOLD_PYTHON" "$ALPHAFOLD_SCRIPT" --json_path=… --run_inference=true`.
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
| **ZDOCK 3.0.2** | `zdock`, `mark_sur`, `create.pl`, `create_lig` (+ legacy `libg2c.so.0`) — **bundled in `dock_bin/`** | `ZDOCK_DIR` env var |
| **distance_antibody** | Lab Fortran helper — source ships in `dock_src/`, compile with gfortran | `DISTANCE_ANTIBODY_BIN` env var |

The pipeline's own Python dependencies (numpy, scipy, aiosqlite, PyMOL) are
installed via conda — see below.

---

## 2. Install

```bash
# 2a. Python environment (pipeline deps only — NOT AlphaFold3)
conda env create -f environment.yml
conda activate pipeline-standalone

# 2b. Build the lab distance helper
gfortran -O2 -o dock_src/distance_antibody dock_src/distance_antibody.f

# 2c. ZDOCK is bundled in dock_bin/ — nothing to install. Use an ABSOLUTE path
#     for ZDOCK_DIR (the pipeline runs from app/, so a relative ./dock_bin breaks).
# 2d. Install AlphaFold3 separately (its own environment / Docker image).
```

---

## 3. Configure

The repo ships a `.env` at its root that is **loaded automatically**. It
contains sample paths from a test machine — **you must edit it for yours**
(use absolute paths). `.env.example` documents every variable; defaults live in
`app/config.py`.

```bash
$EDITOR .env        # edit the committed .env in place
```

The variables you must set:

```bash
# External AlphaFold3 — must be runnable as: $ALPHAFOLD_PYTHON $ALPHAFOLD_SCRIPT ...
ALPHAFOLD_PYTHON=/path/to/af3-venv/bin/python
ALPHAFOLD_SCRIPT=/path/to/alphafold3/run_alphafold.py
ALPHAFOLD_MODEL_DIR=/path/to/alphafold3/models
ALPHAFOLD_DB_DIR=/path/to/alphafold3/databases

# Bundled ZDOCK toolkit — ABSOLUTE path to the repo's dock_bin/
ZDOCK_DIR=/abs/path/to/pipeline-standalone/dock_bin

# Lab Fortran helper built in step 2b — ABSOLUTE path
DISTANCE_ANTIBODY_BIN=/abs/path/to/pipeline-standalone/dock_src/distance_antibody
```

(`mark_sur` needs the legacy `libg2c.so.0`, also bundled in `dock_bin/`; the
pipeline adds `ZDOCK_DIR` to `LD_LIBRARY_PATH` automatically, so no extra setup.)

---

## 4. Run

The quickest start is the bundled known-good example (real 3WZE parameters):

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
| `--alpha` | UCB exploration weight (default `0.1`) |
| `--zdock_workers_per_gpu` | ZDOCK parallelism per GPU (default `10`) |
| `--warmup_size` | Number of random warm-up sequences (default `3 × CDR length`); set small (e.g. `2`) for a quick smoke test |

> The designed chain sequence given to AlphaFold3 is `prefix + CDR + suffix`.
> ZDOCK parallelism is `len(gpu_ids) * zdock_workers_per_gpu`.

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

## Notes

- **GPU indexing**: `--gpu_ids` refers to the GPU ids visible to *your*
  AlphaFold3 process (`CUDA_VISIBLE_DEVICES` is set per worker).
- **Chain labelling**: the designed (CDR) chain is folded under the label from
  `--selected_chain_type` (`L`/`H`) so the distance helper can locate the CDR;
  the fixed chain takes the other label. The same `cdr_chain_id` flows through
  fold → dock → distance, so the CDR is always measured on the right chain.
- **Bundled vs. not**: the ZDOCK toolkit (`dock_bin/`) is included. AlphaFold3
  code, model weights, and databases are **not** — you provide those.
