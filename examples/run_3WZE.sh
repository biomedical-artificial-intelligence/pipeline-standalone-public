#!/usr/bin/env bash
#
# Known-good antibody CDR-design test case (3WZE, Light-Kappa CDR).
# Real production parameters — use this to smoke-test the pipeline end-to-end.
#
# Before running:
#   1. conda activate pipeline-standalone
#   2. cp .env.example .env  &&  edit .env  (AlphaFold3 + ZDOCK paths)
#   3. build dock_src/distance_antibody  (see dock_src/README.md)
#
# Then:   bash examples/run_3WZE.sh
#
# NOTE on smoke-testing: the pipeline first folds a warm-up set of
# (3 x CDR_length) random sequences before the bandit search proper.
# For CDR length 17 that is ~52 folds. To just confirm the wiring works,
# watch output/<pipeline_id>/00001/ produce input_batch_01.pdb ->
# complex_batch_01.pdb and a score in pipeline.log, then Ctrl-C.

set -euo pipefail
cd "$(dirname "$0")/../app"

python run_pipeline.py \
  --pdb_name        ../examples/3WZE.pdb \
  --target_residue  A:1061 \
  --selected_chain_type Light-Kappa \
  --fixed_chain_sequence QVQLVESGGGLVQPGGSLRLSCATSGFTFTDYYMSWVRQPPGKALEWLGFIRNKANGYTTEYSASVKGRFTISRDNSQSILYLQMNTLRAEDSATYYCARDGSYAMDYWGQGTSVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEP \
  --prefix          ELVMTQTPLSLPVSLGDQASISCRSSQSIVHSNGNTYLEWYLQKPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYYCFQGSHVPRTFGGGTKLEIKRTVA \
  --initial_cdr     APSVFIFPPSDEQLKSG \
  --suffix          TASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRG \
  --gpu_ids         "${GPU_IDS:-0}" \
  --max_iter        "${MAX_ITER:-5}" \
  --warmup_size     "${WARMUP_SIZE:-2}" \
  --alpha           0.1

# Smoke-test defaults above (max_iter 5, warmup 2). Override via env, e.g. a
# fuller run:  MAX_ITER=2000 WARMUP_SIZE=51 bash examples/run_3WZE.sh
# (omit --warmup_size entirely in your own command to get the 3 x CDR-len default).
