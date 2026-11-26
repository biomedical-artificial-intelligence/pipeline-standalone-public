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
DOCK_DIR = os.getenv("DOCK_DIR", "/dock")
DOCK_BIN = os.getenv("DOCK_BIN", os.path.join(DOCK_DIR, "zdock"))
DOCK_PREP_BIN = os.getenv("DOCK_PREP_BIN", os.path.join(DOCK_DIR, "mark_sur"))
# Complex-structure generation (part of the ZDOCK distribution).
DOCK_CREATE_PL = os.getenv("DOCK_CREATE_PL", os.path.join(DOCK_DIR, "create.pl"))
DOCK_CREATE_LIG_BIN = os.getenv("DOCK_CREATE_LIG_BIN", os.path.join(DOCK_DIR, "create_lig"))
# Interface-distance helpers. `distance_antibody` (CDR-region) is used by the
# antibody pipeline; its Fortran source ships in this repo (recompile with
# gfortran). `distance` is the peptide variant, kept for completeness.
DISTANCE_ANTIBODY_BIN = os.getenv("DISTANCE_ANTIBODY_BIN", os.path.join(DOCK_DIR, "distance_antibody"))
DISTANCE_BIN = os.getenv("DISTANCE_BIN", os.path.join(DOCK_DIR, "distance"))

# Scratch directory for per-job ZDOCK filtering work (created/removed at runtime).
SCRATCH_DIR = os.getenv("SCRATCH_DIR", "/tmp/dock_scratch")


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
#   FOLD_PYTHON : python interpreter of YOUR AlphaFold3 environment
#   FOLD_SCRIPT : path to run_alphafold.py
#   FOLD_MODEL_DIR / FOLD_DB_DIR : AF3 model weights / databases
# ---------------------------------------------------------------------------
FOLD_PYTHON = os.getenv("FOLD_PYTHON", "python")
FOLD_SCRIPT = os.getenv("FOLD_SCRIPT", "/alphafold/run_alphafold.py")
FOLD_MODEL_DIR = os.getenv("FOLD_MODEL_DIR", "/alphafold/model")
FOLD_DB_DIR = os.getenv("FOLD_DB_DIR", "/alphafold/data")
