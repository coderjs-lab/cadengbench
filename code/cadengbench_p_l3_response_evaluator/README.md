# CADEngBench-P L3 Response Evaluator

This is a separate calibrated evaluator for structural-response fidelity and benchmark-assigned
engineering requirements. It consumes the frozen CADEngBench-P L3 v1 cards and invokes that solver
implementation as a dependency. It does not modify or regrade L3 v1.

The experiment performs actual matched reference/candidate CalculiX solves under one calibrated
fixed resultant force. It reports:

- numerical simulation validity;
- reference-relative p95 von Mises stress, normalized displacement, and normalized compliance;
- stress allowable, displacement, displacement-ratio, and compliance-ratio requirements;
- one decomposed row per nominal or perturbed variant.

All scenarios are benchmark-assigned and set `source_intent_claimed: false`. Results do
not certify real service loading, safety, materials, or standards compliance.

## Commands

```bash
uv sync --extra dev
uv run cadengbench-p-l3-response doctor --ccx "$CADENGBENCH_CCX"

uv run cadengbench-p-l3-response calibrate \
  --profile ../configs/l3_response/cadengbench_p_l3_response_v1.json \
  --card /path/to/physics_card.solver_qualified.json \
  --reference-step /path/to/reference.step \
  --mesh-study /path/to/mesh_sensitivity_summary.json \
  --ccx "$CADENGBENCH_CCX" \
  --output /path/to/calibration

uv run cadengbench-p-l3-response evaluate-pair \
  --contract /path/to/calibration/response_contract.json \
  --card /path/to/physics_card.solver_qualified.json \
  --reference-step /path/to/reference.step \
  --candidate-step /path/to/candidate.step \
  --variant-id nominal \
  --ccx "$CADENGBENCH_CCX" \
  --output /path/to/pair
```

The output contains preflight and calibrated solver evidence, geometry-specific force-controlled
cards, `pair_result.json`, `row.csv`, an evidence manifest, and a final result digest.

`bind-family` freezes matched reference/candidate STEP hashes into a family manifest, and
`run-family` executes that manifest with bounded workers and emits `family_result.json` plus
`results.csv`. Upstream rebuild failures remain typed, non-scoreable rows.

## Pinned HPC Verification

From the repository root on the configured HPC host:

```bash
bash hpc/slurm/cadengbench_p_l3_response/setup_env.sh
sbatch hpc/slurm/cadengbench_p_l3_response/verify.sbatch
sbatch hpc/slurm/cadengbench_p_l3_response/pilot_controls.sbatch
```

The measured development pilot is stored at
`output/cadengbench_p_l3_response_pilots/pilot_summary.json`. The calibrated profile is frozen for
benchmark use, but the results are not safety or source-intent claims.
