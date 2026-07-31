# CADEngBench-A

This directory contains the active assembly benchmark implementation used by
the submission package:

```text
A0 integrity -> A1 grounded joint/entity retrieval -> A2 kinematic verification
```

The earlier exploratory assembly-graph stage is not part of this package, is
not scored, and is not required by any active runtime path. A2 is the current
name for conditional joint-frame prediction and PyBullet verification.

The end-to-end runner is `scripts/run_cadengbench_a_benchmark.py`. The included
development sample is at `../../datasets/cadengbench_a_sample7_v2` and contains
one item from each supported joint family.

From the supplement root, package a no-cost dry run:

```bash
.venv-a/bin/python code/cadengbench_a/scripts/run_cadengbench_a_benchmark.py \
  --dataset-root datasets/cadengbench_a_sample7_v2 \
  --labels-root datasets/cadengbench_a_sample7_v2/labels \
  --provider openai \
  --model dry-run-model \
  --dry-run \
  --output-dir outputs/a_dry_run
```

Run the deterministic A2 physics fixture with:

```bash
.venv-a/bin/python code/cadengbench_a/scripts/run_cadengbench_a_a2_fixture_smoke.py
```
