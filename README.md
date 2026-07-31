# CADEngBench Code and Data Supplement

This anonymous package contains executable benchmark code and compact,
development-only data for both submission tracks:

- **CADEngBench-P:** zero-to-CAD and source-native functional edits with L0-L2
  deterministic scoring and inline CalculiX L3 response-fidelity records for
  every included object.
- **CADEngBench-A:** A0 integrity, A1 grounded joint/entity retrieval, and A2
  conditional joint-frame plus PyBullet kinematic verification.

The discarded exploratory assembly-graph stage is intentionally absent. A2 is
the active kinematic layer; there is no scored A3 in this package.

## Included data

- `datasets/cadengbench_p_sample10_v1`: ten development items, comprising six
  BenchCAD and four Fusion-derived items across easy, medium, and hard cases;
  all ten include solver-qualified L3 annotations and calibrated contracts.
- `datasets/cadengbench_a_sample7_v2`: seven development items, one for each
  supported joint family, with development labels for evaluator reproduction.

These are representative subsets, not the sealed full 300-item P or 150-item A
corpora. They reproduce the complete execution interfaces and sample-level
scores. Full-paper aggregates require the full frozen corpora and preserved
model responses; a new provider call is not guaranteed to reproduce identical
text because hosted inference can change or be nondeterministic.

## Quick start

Python 3.11 is required. Ubuntu/Linux x86-64 is the reference platform. On
Apple Silicon, `micromamba` or `conda` is also required because PyPI does not
publish a PyBullet wheel for that platform. The setup script installs the
Python stacks, CalculiX 2.21, and runs a real solver smoke test. On Ubuntu it
uses conda/micromamba when available or `apt-get` with sudo otherwise.

```bash
bash scripts/setup.sh
python3 scripts/verify_package.py
bash scripts/run_controls.sh
cp .env.example .env
```

Add the API key and model selection to `.env`, then run both tracks:

```bash
bash scripts/run_sample.sh
```

Outputs are isolated under `outputs/<run-id>/`. Raw provider responses,
extracted artifacts, per-item evaluator records, aggregate JSON, A2 simulation
records, and L3 solver results are preserved separately.

To use an existing CalculiX installation, set `CADENGBENCH_CCX` before setup.
For Python-only development, set `CADENGBENCH_SKIP_SYSTEM_DEPS=1`; L3 will then
remain unavailable until the solver installer and smoke test pass.

## Layout

```text
code/                 P and A runners/evaluators
configs/              model and L3 profiles
datasets/             representative development subsets
docs/                 rules, licensing, and reproduction details
reference_results/    compact provider-free reference summaries
schemas/              runtime JSON schemas, including Fusion patch scoring
scripts/              setup, verification, controls, execution, archive build
```

See `docs/REPRODUCIBILITY.md` for exact commands and
`docs/AAAI27_COMPLIANCE.md` for the packaging boundary.
