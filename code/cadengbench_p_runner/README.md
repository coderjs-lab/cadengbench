# CADEngBench-P Candidate Runner

This package materializes frozen CADEngBench-P model requests, calls candidate
models through a provider adapter, and produces immutable submission bundles for
`cadengbench_p_evaluator`. Generation and scoring are separate operations: the
evaluator never imports provider SDKs, reads API keys, or repeats a paid request.

The first release implements the OpenAI-compatible chat protocol. It supports
hosted endpoints and local vLLM servers through the same model registry entry.
Version 0.5.3 supports both zero-to-CAD and source-native functional-edit request
materialization. The CalculiX L3 bridge is restricted to zero-to-CAD until a
certified post-edit native-reference path is available.

Fusion functional-edit requests carry a compact, deterministic index of every
replay-supported extrusion-distance and profile-circle-radius JSON pointer. The
full source remains evaluator-held and hash-bound, so request size stays bounded
without using a lossy CadQuery proxy or weakening non-target preservation.

## Contracts

- `CanonicalModelRequestV1` freezes the system prompt, user prompt, ordered
  visual inputs, response interface, and every content hash before inference.
- `CandidateSubmissionV1` binds one extracted `submission.py` to its request,
  raw response, model identity, extraction method, and successful attempt.
- `CanonicalModelRequestV2` and `CandidateSubmissionV2` add source-native
  BenchCAD Python edits and bounded Fusion 360 JSON patches without weakening
  the frozen v1 zero-to-CAD contract.
- `attempts.jsonl` is append-only and hash-chained. Each provider attempt is
  durable before a retry is allowed.

## Quick start

```bash
uv sync --extra dev

uv run cadengbench-p-run plan \
  --items-root ../benchmarks/cadengbench-p/evalsets/l0-l2-v3/items \
  --model-spec configs/models/local_vllm.example.json \
  --output /tmp/cadengbench-run \
  --run-id example-vllm-run \
  --input-profile text_iso

uv run cadengbench-p-run run --run /tmp/cadengbench-run
uv run cadengbench-p-run status --run /tmp/cadengbench-run
uv run cadengbench-p-run package --run /tmp/cadengbench-run
```

The publication workflow runs and resumes both protocols, then calls the
deterministic evaluator and aggregation stages:

```bash
uv run cadengbench-p-run preflight \
  --evalset ../benchmarks/cadengbench-p/evalsets/l0-l2-v3 \
  --model-spec configs/models/local_vllm.example.json \
  --output ../runs/cadengbench-p/example-vllm-run \
  --run-id example-vllm-run \
  --evaluator-cli ../cadengbench_p_evaluator/.venv/bin/cadengbench-p-eval \
  --input-profile text_iso
```

Preflight performs no inference. It validates the complete evaluator release,
materializes all 300 requests for each protocol, verifies every visual digest,
and writes `benchmark_plan.json`. `benchmark` repeats this idempotent gate before
the first model call and refuses a changed release, model spec, prompt profile,
run ID, or request manifest on resume.

```bash
uv run cadengbench-p-run benchmark \
  --evalset ../benchmarks/cadengbench-p/evalsets/l0-l2-v3 \
  --model-spec configs/models/local_vllm.example.json \
  --output ../runs/cadengbench-p/example-vllm-run \
  --run-id example-vllm-run \
  --evaluator-cli ../cadengbench_p_evaluator/.venv/bin/cadengbench-p-eval \
  --repo-root .. \
  --input-profile text_iso \
  --workers 8
```

The same command is available as `scripts/slurm/run_l0_l2_benchmark.sbatch`.
It is idempotent: generation resumes from the attempt ledger and scoring resumes
from hash-bound per-item results.

Use `resume` after interruption. It has the same execution semantics as `run`,
but communicates operator intent in logs and scripts:

```bash
uv run cadengbench-p-run resume --run /tmp/cadengbench-run
```

The API key value is read only from the environment variable named by
`api_key_env`; it is never written to the run manifest. Local vLLM endpoints may
set `api_key_required` to `false`. The Slurm wrapper optionally loads the
git-ignored `cadengbench_p_runner/.env`; set `CADENGBENCH_ENV_FILE` to use a
different secret file.

Hosted reasoning models may use `max_completion_tokens` and
`reasoning_effort` in `request_defaults`. Keep `max_tokens` for compatible local
servers; a model specification must not set both token-limit fields.

For the official `n=1` setting, evaluator-ready programs are written as
`artifacts/<item_id>/submission.py`. They can be scored directly:

```bash
uv run --project ../cadengbench_p_evaluator cadengbench-p-eval evaluate \
  --evalset /path/to/prepared-evalset \
  --submissions /tmp/cadengbench-run/artifacts \
  --output /tmp/cadengbench-score \
  --model-id example-vllm-run \
  --workers 4
```

For a complete resumable model run, one command generates and scores both frozen
L0-L2 protocols:

```bash
uv run cadengbench-p-run benchmark \
  --evalset /path/to/prepared-300-item-evalset \
  --model-spec configs/models/my-model.json \
  --output /path/to/output/model-run \
  --run-id my-model \
  --evaluator-cli ../cadengbench_p_evaluator/.venv/bin/cadengbench-p-eval \
  --repo-root .. \
  --input-profile text_iso \
  --workers 4
```

Its `benchmark_summary.json` exposes the zero-to-CAD scoring directory as the
L3 handoff. On HPC, submit the conditional physics layer with:

```bash
../hpc/slurm/cadengbench_p_benchmark/submit_l3.sh \
  /path/to/output/model-run/zero_to_cad/scoring my-model
```

## Failure semantics

- Only transport failures, timeouts, HTTP 408/409/425/429, and selected 5xx
  responses are retryable.
- Authentication, invalid request, unsupported modality, and extraction errors
  are terminal for that attempt policy.
- Raw provider envelopes are retained even when extraction fails.
- A model response that violates the output contract is a settled benchmark
  outcome: its raw text is preserved for deterministic L0 failure scoring.
- `run_settled=true` means every request has a success or terminal model/transport
  outcome. `valid_complete_run=true` is stricter and means every response yielded
  an evaluator-ready artifact.
- Re-running never calls the model for a job with a recoverable successful raw
  response.
- The local thread scheduler is bounded by the frozen model concurrency and RPM
  limits. It is intentionally not a distributed scheduler.

## L3 bridge

The runner also composes a completed L0-L2 result directory with the standalone
CalculiX response evaluator. Public commands are `validate-release`, `l3-index`,
`l3-plan`, `l3-run`, and `l3-summarize`. The canonical HPC entry point is
`../hpc/slurm/cadengbench_p_benchmark/l3_from_l02.sbatch`; its output contract is
documented at `../benchmarks/cadengbench-p/evaluators/l3/README.md`.

The default HPC profile is `extended_v1`. It dispatches the frozen 120
surface-traction cards to the unchanged force-controlled evaluator and 44
single-support body-acceleration cards to the fixed-acceleration evaluator.
Both paths consume parameter-matched L2 reference/candidate STEP files and
write the same item, physical-pair, protocol-membership, and aggregate tables.
All 120 surface and 44 body response contracts are precompiled under the pinned
HPC runtime; official model runs do not perform calibration.
Set `CADENGBENCH_L3_PROFILE_MODE=frozen_v1` only when reproducing the original
120-card release.
