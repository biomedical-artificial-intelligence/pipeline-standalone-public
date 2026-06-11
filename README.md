# Antibody CDR Design Pipeline (standalone)

A file-based command-line pipeline that designs antibody CDR sequences by
iteratively folding candidate antibodies with **AlphaFold3**, docking them
against a target antigen with **ZDOCK**, and steering the search with a
**UCB bandit**.

This is a **standalone** build: it does **not** bundle AlphaFold3 and runs no
web server. You install AlphaFold3 yourself and point the pipeline at it via
environment variables. Everything runs from the command line and reads/writes
plain files plus a small SQLite job database.

```
bandit (propose CDR)  ->  AlphaFold3 (fold 2-chain antibody)  ->  ZDOCK (dock vs antigen)  ->  score  ->  bandit
```

---

## 1. Requirements

You provide three external pieces (none are bundled here):

| Component | What | How the pipeline finds it |
|-----------|------|----------------------------|
| **AlphaFold3** | Your own AF3 install (code + model weights + databases) | `ALPHAFOLD_*` env vars |
| **ZDOCK 3.0.2** | `zdock`, `mark_sur`, `create.pl`, `create_lig` | `ZDOCK_DIR` env var |
| **distance_antibody** | Lab Fortran helper — source ships in `dock_src/` | `DISTANCE_ANTIBODY_BIN` env var |

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

# 2c. Install ZDOCK 3.0.2 separately and note its directory.
# 2d. Install AlphaFold3 separately (its own environment).
```

---

## 3. Configure

Copy the template and edit paths for your machine:

```bash
cp .env.example .env
$EDITOR .env
```

Key variables (full list in `.env.example`, defaults in `app/config.py`):

```bash
# External AlphaFold3 (runs in ITS OWN environment)
ALPHAFOLD_PYTHON=/path/to/af3-venv/bin/python
ALPHAFOLD_SCRIPT=/path/to/alphafold3/run_alphafold.py
ALPHAFOLD_MODEL_DIR=/path/to/alphafold3/models
ALPHAFOLD_DB_DIR=/path/to/alphafold3/databases

# ZDOCK toolkit (zdock, mark_sur, create.pl, create_lig)
ZDOCK_DIR=/path/to/zdock

# Lab Fortran helper built in step 2b
DISTANCE_ANTIBODY_BIN=/abs/path/to/dock_src/distance_antibody
```

`.env` is loaded automatically from the repo root.

---

## 4. Run

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
  the fixed chain takes the other label.
- No AlphaFold3 weights, ZDOCK binaries, or databases are included in this
  repository.
