# CADEngBench-P Real Evaluator Development Slice

This directory contains 10 CADEngBench-P parts used to calibrate and
regression-test the deterministic L0-L3 benchmark path.

## Scope

- 6 BenchCAD and 4 Fusion 360 Reconstruction items.
- Easy, medium, and hard items, all solver-qualified for L3.
- Single-solid and multi-solid geometry.
- Bounding-box and cylindrical-hole parameter observables.
- Two items with a certified development-only minimum-hole DFM contract.
- One positive equivalent CadQuery control and one semantic parameter mutant per
  item.
- Complete L3 annotation, provenance, solver-qualified physics-card,
  source-qualification, and calibrated response-contract records under each
  `items/<item_id>/l3/` directory.

The core item records come from the final 300-item release. The supplement adds
evaluator-owned L0-L2 qualification and frozen L3 records directly to each
selected item directory. Qualifications freeze tolerances, parameter domains,
measurement bindings, DFM applicability, physics probes, and response contracts.

## Reproduce

Run from the supplement root after `bash scripts/setup.sh`:

```bash
bash scripts/run_controls.sh
```

The packaged controls require all 10 positive submissions to pass L0-L2 and
the semantic mutants to trigger the frozen development-slice failure checks.
The same command runs a CalculiX smoke test and an L3 reference control for all
10 objects. Raw meshes, solver decks, and field files are regenerated during
the run and are intentionally excluded from the size-limited supplement.

The DFM family definitions, authority gates, and verdict logic are unchanged.
The wall estimator's sampled opposing-surface fallback now uses oriented trimmed
B-Rep samples, so no active L1 path invokes tessellation.

The Fusion controls are evaluator calibration fixtures generated from supported
native-history profiles. They are not replacement ground truth and are not
distributed as model answers.
