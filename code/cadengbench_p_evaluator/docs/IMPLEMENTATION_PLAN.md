# Implementation Status

Date: 2026-07-22

## Implemented

1. Standalone `uv` package, frozen profiles, JSON schemas, CI, and Apptainer recipe.
2. Hash-bound compiler for the real BenchCAD/Fusion 30-item canary.
3. L0 interface/policy checks, subprocess execution, valid-solid checks, and STEP round trip.
4. L1 candidate-local B-Rep analysis, typed engineering requirements, isolated
   geometry processing, and the two DFM families. Canonical-reference similarity
   is retired from active scoring.
5. L2 exact independent/derived parameter bindings, qualified domains,
   feasibility-aware directed/real-Morris/coupled schedules, invalid probes,
   variant L0/L1/DFM replay, response analysis, typed relations, and sound Z3 proofs.
6. Deterministic functional target/effect/preservation kernel, revised-CadQuery
   BenchCAD and bounded-JSON/OCC Fusion adapters, post-edit replay, batch CLI, and resume.
7. Exact run rosters, item concurrency, evidence manifests, commit digests,
   interrupted-item recovery, strict resume, standalone aggregation validation,
   and environment/build provenance.
8. Unit, mutation, synthetic integration, integrity, and real-canary compiler tests.
9. L3 sidecar preparation for the frozen 300-item roster, including deterministic
   eligibility evidence, one-model annotation provenance, and fail-closed card compilation.
10. Linux/HPC-only Gmsh C3D10 meshing, CalculiX deck compilation/execution, FRD parsing,
    equilibrium and small-deformation gates, regularized response metrics, VTU/PNG output,
    source-card qualification, mesh-sensitivity studies, and L2-family aggregation.
11. Canonical axial and cantilever verification cases. Both pass analytic tolerances; the
    tests also detected and corrected the Gmsh-to-CalculiX C3D10 midside-node permutation.

## Not publication-ready

| Blocker | Current evidence | Required action |
|---|---|---|
| Requirement-ledger coverage | Active L1 no longer relies on canonical-reference similarity | Regrade controls and certify each requirement family with equivalent positives and semantic mutants |
| L2 data authority | The 10-item development slice has complete qualified domains and schedules | Scale qualification beyond the slice and freeze public/private contract disclosure |
| DFM authority | Two minimum-hole cases are scoreable; the remaining checks are explicitly not applicable | Expand only with certified applicability, extraction, and public process thresholds |
| Functional Protocol E | Both source adapters and execution paths are implemented; current candidate effect text is not scoring authority | Promote independent deterministic effect oracles and expand mutants/coverage |
| Adversarial isolation | macOS backend is subprocess isolation only | Run official scoring in the frozen Linux container with scheduler resource limits |
| Runtime calibration | Development budgets are provisional | Freeze worker count, geometry budget, and per-item L2 variant budget from calibration data |
| L3 eligible subset | 214/300 pass deterministic prefilter; 196 cards compile and 120 pass source qualification with zero evaluator errors | Report source/difficulty applicability before conditional results; retain all exclusion stages |
| L3 numerical calibration | Canonical mechanics, three cross-source mesh studies, and two five-instance family controls pass | Expand matched-family calibration only if response preservation is proposed as a paper metric |
| L3 response thresholds | Stress and compliance are diagnostic under a standardized probe | Do not add safety/allowable claims; calibrate reference-response preservation separately if matched variants become available |

## Next gated slice

The next L0-L2 gate is to recompile and regrade the real development slice under
ADR-0030, then expand requirement-ledger and functional-edit mutants. Protocol E
implementation is complete; independently measurable engineering-effect coverage
remains a dataset-authority task. L3 proceeds independently through the bounded
production path below.

## L3 bounded production path

L3 v1 does not expand analysis breadth to create nominal coverage. It uses one benchmark-assigned
linear-static response probe with fixed `E=70000 MPa`, `nu=0.33`, and `0.1 MPa` surface traction.
Only `axial_tension` and `cantilever_transverse` are permitted. The single frozen VLM chooses a
candidate region pair and scenario from deterministic geometry evidence; it cannot set numerical
physics, tolerances, checks, or promotion state.

The completed initial sequence was:

1. Retain exactly one annotation attempt for each of the 300 sidecar bundles.
2. Compile only schema-valid and geometrically consistent proposals: 196 cards.
3. Run three cross-source real-item pilots with mesh sensitivity and inspectable von Mises fields.
4. Run every compiled card against its bound source STEP and promote only complete solver passes:
   120 cards.
5. Bind promoted cards to existing L0-L2 positive-control families and verify the public `l3.json`
   handoff before any benchmark model run.

Buckling, thermal, modal, pressure, contact, nonlinear, fluid, shell, assembly, real-material
allowables, and service-load claims are explicitly outside L3 v1.
