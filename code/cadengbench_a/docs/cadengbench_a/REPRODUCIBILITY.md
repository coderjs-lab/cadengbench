# CADEngBench-A Reproducibility Checklist

## Frozen inputs

- [x] Core-150 contains 30 development and 120 held-out test items.
- [x] Every public file is listed by path, byte count, and SHA-256.
- [x] Every evaluator label is separately hashed.
- [x] Public items contain no `master.json` or ground-truth directory.
- [x] A1 and A2 prompts and JSON schemas are versioned.
- [x] A seven-family development sample is included for artifact review.

Verify the full local release:

```bash
python scripts/verify_cadengbench_a_release.py \
  artifacts/cadengbench_a_core150_public_v1 \
  --labels-root artifacts/cadengbench_a_core150_v1/sealed_ground_truth
```

Verify the compact submission sample:

```bash
python scripts/verify_cadengbench_a_release.py \
  submission/cadengbench_a_sample7_v2
```

## Environment

```bash
conda env create -f environment-cadengbench-a.yml
conda activate cadengbench-a
cp .env.example .env
```

Only `.env` contains provider credentials, and it is excluded from version control.

## No-cost preflight

```bash
python scripts/run_cadengbench_a_benchmark.py \
  --dataset-root submission/cadengbench_a_sample7_v2 \
  --labels-root submission/cadengbench_a_sample7_v2/labels \
  --provider openai --model dry-run-model --dry-run \
  --output-dir outputs/sample7_preflight
```

The preflight performs A0 and resolves the exact A1 evidence package without making
an API request. Validate all A2 simulator mappings locally with:

```bash
python scripts/run_cadengbench_a_a2_fixture_smoke.py
```

## Execution invariants

- The same ordered public images, text evidence, prompt, and output schema are sent
  to each provider.
- Hidden labels are opened only after model output is archived.
- A2 consumes the model's rank-1 A1 hypothesis; labels never repair it.
- PyBullet executes only a schema-valid A2 pose that passes the frozen frame gates.
- Invalid, unsupported, and not-scored outcomes remain explicit.
- `--resume` reuses completed item archives instead of issuing duplicate calls.

## Archived outputs

Each run records model/provider identity, exact prompts, image hashes, raw response,
parsed prediction, parser diagnostics, A1 metrics, A2 metrics and simulation result,
latency, and token usage. `run_manifest.json` is the machine-readable run index.

## Release boundary

The compact development sample may be included in supplementary material. The full
public Core-150 input package should be distributed as a release artifact because it
is approximately 245 MB. Held-out test labels remain evaluator-side and must never
be published with model inputs.
