# CADEngBench-P Evaluator

Standalone deterministic evaluator for CADEngBench-P Protocol Z (zero-to-CAD)
and Protocol E (source-native functional edits). Protocol E accepts revised
CadQuery for BenchCAD and bounded JSON patches for Fusion, replayed through the
independent JSON-to-OCC backend.

## Public hierarchy

- **L0:** exact submission interface, isolated execution, valid solids, and STEP round trip.
- **L1:** candidate-local B-Rep facts, typed nominal engineering requirements,
  applicable certified DFM, and non-scoring reference-geometry diagnostics.
  Similarity becomes score-bearing only for an explicit exact-reconstruction task.
- **L2:** independent/derived parameter binding, qualified domains,
  directed/Morris/coupled/invalid probes, per-variant L0/L1/DFM replay,
  deterministic response and relation checks, sound Z3 obligations, and
  functional target/effect/preservation plus post-edit robustness.
- **L3:** an eligible-subset standardized linear-static CalculiX response probe over the nominal
  model and L2-passing variant family.

CADEngBench-A is intentionally absent. Official scoring never invokes an LLM or VLM; the separate
L3 sidecar uses one frozen VLM only during offline card construction, before deterministic source
qualification.

## Quick start

```bash
uv sync --extra dev
uv run cadengbench-p-eval doctor
uv run cadengbench-p-eval prepare-case \
  --item-root ../data/interim/cadengbench_p_r25_combined_hpc_canary_30/dataset/items/p_benchcad_000078 \
  --profile configs/profiles/dev.json \
  --output /tmp/p_benchcad_000078.case.json
uv run pytest
```

Evaluate one model submission:

```bash
uv run cadengbench-p-eval evaluate-item \
  --case /tmp/p_benchcad_000078.case.json \
  --submission /path/to/submission.py \
  --output /tmp/p_benchcad_000078.result
```

Evaluate a prepared set with item-level process parallelism:

```bash
uv run cadengbench-p-eval evaluate \
  --evalset /path/to/prepared-evalset \
  --submissions /path/to/model-submissions \
  --output /path/to/run \
  --model-id model-and-decoding-config-v1 \
  --workers 4
```

Prepare and evaluate source-native functional edits:

```bash
uv run cadengbench-p-eval prepare-evalset \
  --items-root /path/to/dataset/items \
  --profile configs/profiles/paper.json \
  --protocol functional_edit \
  --output /path/to/prepared-functional

uv run cadengbench-p-eval evaluate \
  --evalset /path/to/prepared-functional \
  --submissions /path/to/edit-submissions \
  --repo-root .. \
  --output /path/to/functional-run \
  --model-id model-and-decoding-config-v1 \
  --workers 4
```

Build the frozen 10-real-part evaluator development slice:

```bash
uv run cadengbench-p-eval build-dev-slice \
  --items-root ../data/eval/cadengbench_p_dev_slice_10_v1/items \
  --plan ../data/eval/cadengbench_p_dev_slice_10_v1/slice_plan.json \
  --output ../data/eval/cadengbench_p_dev_slice_10_v1

uv run cadengbench-p-eval prepare-evalset \
  --items-root ../data/eval/cadengbench_p_dev_slice_10_v1/items \
  --profile configs/profiles/dev_slice_10.v2.json \
  --output ../data/eval/cadengbench_p_dev_slice_10_v1/prepared_brep_v2p2
```

Production items use `benchmark/consumer_ready_item_v2.json` plus a separate,
hash-bound `evaluation/qualification.v2.json`. The latter owns evaluator policy
such as tolerances, qualified L2 domains, direct measurement bindings, and DFM
applicability; it does not modify dataset annotations.

Candidate API/vLLM generation is intentionally outside this package. The
separate `../cadengbench_p_runner/` package freezes model requests and writes its
official `n=1` artifacts as `artifacts/<item_id>/submission.py`; pass that
`artifacts/` directory to `--submissions`. Regrading never requires a provider
SDK or API key.

Use `--resume` for an interrupted batch. A committed item is reused only when its
case, profile, submission, evaluator, environment, result digest, and complete
evidence manifest still match. An uncommitted item directory is removed and
recomputed from its item boundary.

## Authority rules

- A hard check uses only frozen deterministic evidence.
- Missing required authority is `not_scoreable`, not a pass.
- Infrastructure failure is `evaluator_error`, not a model failure.
- DFM is score-bearing only under certified extraction and explicit thresholds.
- L2 refuses to replay unless every public parameter has an explicit binding,
  qualified domain, requirement-update contract, and schedule coverage.
- The geometry fingerprint is kernel/profile-frozen response evidence, not a
  representation-independent geometry equivalence test.

L1 loads the candidate STEP once and records its B-Rep facts, deterministic
fingerprint, feature inventory, and bound parameter measurements. Only typed
engineering requirements and applicable DFM predicates affect its verdict.
Chamfer distance, normalized Chamfer, surface precision/recall/F1, HD95, voxel
IoU/Dice, volume and area error, centroid error, and bounding-box error are
reported as non-scoring diagnostics on nominal engineering-specification tasks.
They support comparison with prior CAD benchmarks and do not alter pass/fail.

For a separately identified `exact_reconstruction` task, the case must freeze
`reference_similarity_thresholds`; only then does the reference comparison add a
required L1 gate. L2 parameter variants and functional edits are not compared to
the nominal reference because it is not their intended post-edit geometry.

The current zero-to-CAD prompt contract does not define world X/Y/Z axes.
Consequently, global orientation is not score-bearing: envelope dimensions are
matched as a tolerance-aware unordered set, and L2 envelope parameters bind to
the matching candidate dimension. Fixed-axis checks remain available only for
future tasks that expose the required frame in the public prompt.

Likewise, prompt-v2 cylindrical-opening checks score only publicly disclosed
quantity and diameter. Source-native depth, axis, and center values remain
provenance unless a future public prompt explicitly exposes them. A failed L2
variant marks dependent response and relation checks `not_reached`; it remains
a model failure and does not turn a complete run into an evaluator-invalid run.

The v3 engineering policy allows the larger of 0.1 mm and 1% error for
continuous envelope and opening dimensions. Validity, body and feature counts,
submission execution, parameter sensitivity, functional-edit effects, and
preservation remain semantic or exact gates. Aggregate results retain strict
hierarchical pass rates and additionally report layer reach rate, conditional
pass rate among reached items, and mean required-check progress.

## L3 HPC workflow

L3 is Linux/HPC-only and uses a separate environment and data sidecar:

```bash
bash ../hpc/slurm/cadengbench_p_l3/setup_env.sh

cadengbench-p-eval l3-prepare-annotations \
  --repo-root .. \
  --profile ../configs/l3/cadengbench_p_l3_probe_v1.json

cadengbench-p-eval l3-annotate \
  --repo-root .. \
  --profile ../configs/l3/cadengbench_p_l3_probe_v1.json

cadengbench-p-eval l3-compile-cards \
  --repo-root .. \
  --profile ../configs/l3/cadengbench_p_l3_probe_v1.json

cadengbench-p-eval l3-qualify-corpus \
  --repo-root .. \
  --profile ../configs/l3/cadengbench_p_l3_probe_v1.json \
  --output ../data/derived/cadengbench_p_300_l3_probe_v1 \
  --ccx "$HOME/.local/bin/ccx" \
  --workers 3
```

The Slurm entry points under `../hpc/slurm/cadengbench_p_l3/` run the single-model annotation,
source-card qualification, and real-item visualization pilots. `l3-build-family` binds one promoted
card to an existing L0-L2 item output; `l3-run-family` writes detailed variant results plus the
public `l3.json` layer contract.

Source qualification writes the exact Gmsh mesh, CalculiX deck/logs/FRD, parsed metrics, stage
verdict, promoted card, exact evidence manifest, explicit result target, and final commit digest. A
requested visualization writes a true quadratic-tetrahedron VTU plus a display-independent PNG
rendered from subdivided C3D10 boundary faces. Rendering is optional evidence and cannot change a
physics verdict.

Each source card or family variant runs in a fresh short-lived subprocess, so a native Gmsh/OCCT
failure cannot corrupt the corpus worker. The frozen mesh policy does not invoke Gmsh's optional
`HighOrder` optimizer; mesh suitability is enforced by the explicit element-count and minimum-SICN
gates.

The v1 material and traction are standardized probes with `source_intent_claimed: false`. Stress
and compliance are diagnostic; L3 does not claim real service loading, structural safety, material
allowables, or standards compliance.

For orientation-free prompts, L3 estimates one proper rigid transform from the
nominal candidate and applies it unchanged to the complete parameter family.
Axis selectors resolve the outermost aligned planar regions by plane
intersection, so small registration residuals do not require numerically
axis-collapsed faces. No scale, deformation, or geometry repair is permitted.

## Current release blockers

The L0-L2 software paths are implemented for both protocols. All 30 local mixed-
source canary items compile as Protocol-E cases, and real BenchCAD/Fusion golden
edits plus a no-op mutant exercise the adapters. Publication release still
requires requirement-ledger and mutation certification after the L1 reset,
independently promoted functional engineering-effect coverage, broader DFM
authority, frozen runtime budgets, and the official Linux isolation backend.

See [Architecture](docs/ARCHITECTURE.md), [Implementation status](docs/IMPLEMENTATION_PLAN.md),
and [DFM scope](docs/DFM_SCOPE.md).
