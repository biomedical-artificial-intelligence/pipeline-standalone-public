#!/usr/bin/env bash
#
# Peptide-design test case (--mode peptide): design a free single-chain peptide
# against a target, using the SAME UCB bandit + early-stop engine as antibody mode.
#
# Differences from antibody mode:
#   - 1-chain AlphaFold3 fold (peptide only; no fixed chain / prefix / suffix)
#   - distance helper = dock_src/distance (peptide), DISTANCE_MAX = 15 A
#   - --initial_cdr is the SEED PEPTIDE (its length = design length)
#
# Before running:
#   1. conda activate pipeline-standalone
#   2. cp .env.example .env  &&  edit .env  (AlphaFold3 + ZDOCK + DISTANCE_BIN paths)
#   3. build dock_src/distance        (gfortran -O2 -o dock_src/distance dock_src/distance.f)
#
# Then:   bash examples/run_peptide_3WZE.sh
#
# NOTE: the target/seed below are a TEMPLATE for wiring smoke-tests — replace the
# target residue and seed peptide with your real case.

set -euo pipefail
cd "$(dirname "$0")/../app"

python run_pipeline.py \
  --mode            peptide \
  --pdb_name        ../examples/3WZE.pdb \
  --target_residue  A:1061 \
  --initial_cdr     APSVFIFPPS \
  --gpu_ids         "${GPU_IDS:-0}" \
  --max_iter        "${MAX_ITER:-5}" \
  --warmup_size     "${WARMUP_SIZE:-2}" \
  --alpha           0.1

# Smoke-test defaults (max_iter 5, warmup 2). Fuller run:
#   MAX_ITER=2000 WARMUP_SIZE=30 bash examples/run_peptide_3WZE.sh
# (omit --warmup_size for the 3 x length default.)
