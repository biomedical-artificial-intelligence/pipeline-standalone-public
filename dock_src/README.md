# dock_src — interface-distance helper (lab source)

`distance_antibody.f` is the lab's own Fortran helper that measures the
minimum distance between the receptor target residue and the antibody CDR
region of a docked complex. It is used by the pose-filtering step
(`app/services/dock/filtering.py`).

## Build

```bash
gfortran -O2 -o distance_antibody distance_antibody.f
```

Then point the pipeline at the resulting binary (in `.env`):

```bash
DISTANCE_ANTIBODY_BIN=/abs/path/to/distance_antibody
```

(or place it inside your `ZDOCK_DIR`, which is the default search location).

## Usage (called internally by the pipeline)

```
distance_antibody <complex.pdb> <receptor_chain> <target_res> <cdr_start> <cdr_end> <cdr_chain>
```

It writes the computed distance to `fort.2` in the current working directory.
