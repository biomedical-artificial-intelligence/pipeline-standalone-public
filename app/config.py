"""
Central configuration for the standalone pipeline.

Every host-specific path is read from an environment variable with a sensible
default. To run on your own machine, set the variables below (e.g. in a `.env`
file or your shell) — you should not need to edit any source file.

See `.env.example` in the repository root for a template.
"""
import os

# Load a `.env` file from the repository root if present, so users can keep
# their machine-specific paths in one file instead of exporting variables.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass


# ---------------------------------------------------------------------------
# ZDOCK toolkit  (external binaries — install ZDOCK 3.0.2 separately)
#   zdock     : rigid-body docking
#   mark_sur  : surface marking preprocessor
#   distance  : interface-distance helper used during pose filtering
# ---------------------------------------------------------------------------
ZDOCK_DIR = os.getenv("ZDOCK_DIR", "/dock")
ZDOCK_BIN = os.getenv("ZDOCK_BIN", os.path.join(ZDOCK_DIR, "zdock"))
MARK_SUR_BIN = os.getenv("MARK_SUR_BIN", os.path.join(ZDOCK_DIR, "mark_sur"))
# Complex-structure generation (part of the ZDOCK distribution).
CREATE_PL = os.getenv("CREATE_PL", os.path.join(ZDOCK_DIR, "create.pl"))
CREATE_LIG_BIN = os.getenv("CREATE_LIG_BIN", os.path.join(ZDOCK_DIR, "create_lig"))
# Interface-distance helpers. `distance_antibody` (CDR-region) is used by the
# antibody pipeline; its Fortran source ships in this repo (recompile with
# gfortran). `distance` is the peptide variant, kept for completeness.
DISTANCE_ANTIBODY_BIN = os.getenv("DISTANCE_ANTIBODY_BIN", os.path.join(ZDOCK_DIR, "distance_antibody"))
DISTANCE_BIN = os.getenv("DISTANCE_BIN", os.path.join(ZDOCK_DIR, "distance"))

# Scratch directory for per-job ZDOCK filtering work (created/removed at runtime).
SCRATCH_DIR = os.getenv("SCRATCH_DIR", "/tmp/zdock_scratch")


# ---------------------------------------------------------------------------
# Pipeline I/O
# ---------------------------------------------------------------------------
# Base directory for input/output. Defaults to the directory containing this
# file (i.e. the `app/` package), so the pipeline works from a fresh git clone.
BASE_PATH = os.getenv("PIPELINE_BASE", os.path.dirname(os.path.abspath(__file__)))
INPUT_BASE_DIR = os.getenv("INPUT_DIR", os.path.join(BASE_PATH, "input"))
OUTPUT_BASE_DIR = os.getenv("OUTPUT_DIR", os.path.join(BASE_PATH, "output"))


# ---------------------------------------------------------------------------
# External AlphaFold3  (user-installed — NOT bundled with this pipeline)
#   ALPHAFOLD_PYTHON : python interpreter of YOUR AlphaFold3 environment
#   ALPHAFOLD_SCRIPT : path to run_alphafold.py
#   ALPHAFOLD_MODEL_DIR / ALPHAFOLD_DB_DIR : AF3 model weights / databases
# ---------------------------------------------------------------------------
ALPHAFOLD_PYTHON = os.getenv("ALPHAFOLD_PYTHON", "python")
ALPHAFOLD_SCRIPT = os.getenv("ALPHAFOLD_SCRIPT", "/alphafold/run_alphafold.py")
ALPHAFOLD_MODEL_DIR = os.getenv("ALPHAFOLD_MODEL_DIR", "/alphafold/model")
ALPHAFOLD_DB_DIR = os.getenv("ALPHAFOLD_DB_DIR", "/alphafold/data")
