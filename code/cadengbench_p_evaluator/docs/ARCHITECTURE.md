# Architecture

```text
dataset item -> hash-bound case -> model response -> exact artifact extraction
                                                    |
                                                    v
                                      L0 isolated CadQuery execution
                                                    |
                                      valid STEP + execution evidence
                                                    |
                                                    v
                                  isolated L1 geometry/requirements/DFM
                                                    |
                                                    v
                              L2 qualified schedules and isolated replays
                                                    |
                                                    v
                          solver-qualified L3 card + L2-passing STEP family
                                                    |
                                                    v
                         Gmsh C3D10 -> CalculiX -> typed response evidence
                                                    |
                                                    v
                     evidence manifest -> result JSON -> result commit digest
```

## Module boundaries

- `case.py` is the only module that reads the dataset layout. It emits a compact
  case whose semantic payload and referenced artifacts are hash-bound. Production
  items compile from the consumer-ready record plus a separate evaluator-owned
  qualification, so scoring policy is not written back into dataset annotations.
- `development.py` deterministically prepares real-item qualifications,
  equivalent-positive controls, and semantic parameter mutants. It is calibration
  tooling and is not imported by public scoring.
- `execution/` validates and executes the CadQuery v1 submission interface in a
  fresh subprocess, then exports and reimports STEP.
- `geometry/` owns STEP import, candidate-local facts, requirements,
  measurements, and a kernel/profile-frozen response fingerprint. The standalone
  reference-metric utility is offline diagnostic tooling and is not imported by
  active L1 scoring. L1 does not construct a mesh.
- `dfm/` contains only minimum wall thickness and minimum cylindrical-hole
  geometry, both gated by explicit authority.
- `parametrics/` owns independent/derived parameter binding, qualified domains,
  directed/Morris/coupled schedules, invalid probes, typed relations, sound Z3
  proof records, replay, response evidence, and functional target/effect/
  preservation predicates.
- `protocols/functional_edit.py` owns source-native revised-CadQuery BenchCAD and
  bounded-JSON-patch Fusion replay adapters.
- `physics/` owns the separate L3 annotation sidecar, deterministic card compiler, source
  qualification, Gmsh/CalculiX template, FRD parsing, visualization, and family runner.
- `layers/` maps these primitives to public L0/L1/L2/L3 outcomes.
- `runner.py`, `integrity.py`, and `aggregation.py` own transactional execution,
  exact run rosters, provenance, resume, and publication-validity checks.

## Status semantics

| Status | Meaning | `passed` |
|---|---|---:|
| `passed` | Predicate ran and held | `true` |
| `failed` | Model output violated a runnable predicate | `false` |
| `not_applicable` | Frozen contract excludes the check | `null` |
| `not_scoreable` | Required authority is absent | `null` |
| `evaluator_error` | Evaluator or infrastructure failed | `null` |
| `not_reached` | An upstream layer did not pass | `null` |

Layer status is the strict reduction of required checks. An evaluator error in a
scoring or required diagnostic path invalidates the item outcome. Requested L3
PNG rendering is presentation-only evidence: its failure is recorded but cannot
change an already computed physics verdict. Aggregation is publication-valid only
for an exact frozen roster whose final outcomes are all `passed` or `failed` and
whose required evidence is intact.

## Isolation and integrity

L0 and every L2 variant run in fresh execution subprocesses. L1 runs in a separate
worker process with timeout and process-group termination. Batch items can execute
concurrently through `ProcessPoolExecutor`.

Every L3 source card and family variant also runs in a fresh short-lived subprocess. Native
Gmsh/OCCT termination therefore becomes a typed item failure rather than a broken shared pool. The
frozen L3 mesh policy uses HXT C3D10 with explicit element-count and minimum-SICN gates and does not
run the optional Gmsh high-order optimizer.

Each completed item is a transaction: all evidence files are listed with size and
SHA-256 in `evidence_manifest.json`; then `result.json` is written; finally
`result.sha256` acts as the commit marker. Resume accepts the item only after
semantic status reduction, identity bindings, exact evidence roster, and all hashes
verify. Standalone aggregation repeats the run-manifest, roster, result, and evidence
checks instead of trusting a directory scan.

The local macOS process backend is a development boundary, not an adversarial
sandbox. Official untrusted evaluation must use the frozen Linux container and
cluster-enforced CPU, memory, filesystem, and network limits. Environment identity
records Python, CAD/OCC/VTK dependencies, `uv.lock`, and an optional container digest.

## Protocol boundary

Protocol Z and Protocol E are integrated end to end and produce the same L0/L1/L2
result envelope. They retain source-authoritative submission languages: CadQuery
for BenchCAD and bounded Fusion JSON patches plus independent OCC replay. No
Fusion CadQuery ground truth is claimed.

## Deliberate limits

- No CADEngBench-A path and no assembly/contact physics.
- No LLM/VLM judge.
- No invented parameter ranges, process thresholds, standards, or relations.
- No claim that the response fingerprint proves arbitrary B-Rep equivalence.
- No nominal canonical-reference shape-equivalence scoring in L1.
- No L1 analysis mesh. The wall-thickness estimator retains its existing
  opposing-surface logic but now consumes oriented trimmed-B-Rep samples; future
  simulation meshing belongs to L3.
- L3 v1 is a benchmark-assigned linear-static response probe, not real
  load/material/safety certification. Buckling, thermal, modal, nonlinear, and source-intent
  requirements are deferred.
- No publication profile until requirement-ledger mutants, eligible DFM
  contracts, parameter-domain budgets, and functional engineering-effect oracles
  are certified.
