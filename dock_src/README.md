# dock_src — interface-distance helpers (lab source)

Two Fortran helpers measure the minimum distance between the receptor target
residue and the docked ligand, used by the pose-filtering step
(`app/services/dock/filtering.py`):

- `distance_antibody.f` — **antibody mode**: distance to the antibody CDR region
  (CDR residue range + CDR chain id).
- `distance.f` — **peptide mode**: distance to the peptide ligand (chain `P`,
  residues `1..peptide_length`).

The two are NOT interchangeable (different ligand-atom bounds + chain handling).

## Build

```bash
gfortran -O2 -o distance_antibody distance_antibody.f   # antibody mode
gfortran -O2 -o distance          distance.f            # peptide mode
```

Then point the pipeline at the resulting binaries (in `.env`):

```bash
DISTANCE_ANTIBODY_BIN=/abs/path/to/distance_antibody    # antibody mode
DISTANCE_BIN=/abs/path/to/distance                      # peptide mode
```

(or place them inside your `DOCK_DIR`, the default search location).

## Usage (called internally by the pipeline)

```
distance_antibody <complex.pdb> <receptor_chain> <target_res> <cdr_start> <cdr_end> <cdr_chain>
distance          <complex.pdb> <receptor_chain> <target_res> 1 <peptide_length> P
```

Each writes the computed distance to `fort.2` in the current working directory.
