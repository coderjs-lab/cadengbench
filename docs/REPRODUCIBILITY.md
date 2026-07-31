# Reproducibility Guide

## Environment

Use Ubuntu/Linux x86-64 and Python 3.11 for the reference full path.
`scripts/setup.sh` creates separate P and A environments because the tracks pin
different CAD/runtime stacks. On Apple Silicon it uses conda-forge for the A
environment because PyPI has no PyBullet wheel for that platform. Gmsh is
installed in the P environment. CalculiX 2.21 is installed automatically and
checked by executing `configs/l3/calculix_smoke.inp`.

The CalculiX installer uses this order:

1. An existing executable specified by `CADENGBENCH_CCX` or available as
   `ccx`.
2. A local `.solver` environment through micromamba or conda.
3. Ubuntu/Debian `calculix-ccx` through `apt-get` when no environment manager
   is available.

The local environments and smoke outputs are excluded from the submission ZIP.
The frozen paper reference is Ubuntu/Linux x86-64 with CalculiX 2.21. Other
supported platforms are useful for reproduction checks but may have different
solver binary hashes.

Package references:

- Ubuntu 24.04 `calculix-ccx`:
  https://packages.ubuntu.com/noble/calculix-ccx
- conda-forge CalculiX builds:
  https://anaconda.org/conda-forge/calculix

## API-free verification

```bash
python3 scripts/verify_package.py
bash scripts/setup.sh
bash scripts/run_controls.sh
```

This validates file hashes, schemas, active A0/A1/A2 scope, P source balance,
all ten inline L3 card/reference bindings, A request packaging, and the deterministic A2
PyBullet fixture without a provider request. Setup also performs a provider-free
CalculiX solve and requires a non-empty FRD result. `run_controls.sh` additionally
evaluates the ten positive and semantic-mutant P controls through L0-L2, compiles
both P protocols, and runs all ten source STEP files through their embedded cards
and contracts as reference-versus-reference L3 controls.

To repeat only the system dependency installation and solver test:

```bash
bash scripts/install_system_dependencies.sh
bash scripts/test_calculix.sh
```

## End-to-end provider run

```bash
cp .env.example .env
# Edit .env without committing or submitting it.
bash scripts/run_sample.sh
```

The command prepares P zero-to-CAD and functional-edit evalsets, materializes
requests, calls the selected model, archives raw responses, extracts candidate
artifacts, runs P L0-L2, runs A0-A2, and runs L3 for every P object with a
valid candidate STEP. It resumes completed work in place.

The default P model spec uses the OpenAI-compatible endpoint. To use another
compatible endpoint, copy `configs/models/openai_gpt_5_2_sample.json`, change
the model, base URL, key environment variable, and rate limits, then set
`CADENGBENCH_P_MODEL_SPEC`. Set the corresponding A provider and model in
`.env`.

## Interpreting reproducibility

Deterministic evaluator replay over the same candidate artifacts should produce
the same verdicts and metrics within the pinned numerical environment. A fresh
hosted-model query cannot guarantee byte-identical responses. The representative
subset reproduces sample-level behavior; it cannot recalculate full 300-item P
or 150-item A paper aggregates without those full frozen corpora.

## Archive build

```bash
python3 scripts/build_archive.py \
  --output ../cadengbench_aaai27_code_data_supplement.zip
```

The builder refreshes sample/package checksums, excludes environments and run
outputs, emits deterministic ZIP metadata, and rejects archives at or above
50 MiB.
