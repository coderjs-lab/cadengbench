# CADEngBench-P L3 Extended Evaluator

This package is the versioned extension to the frozen CADEngBench-P L3 v1
implementation. It does not modify the core 300-item release, the frozen
120-card applicability index, or the calibrated response-fidelity evaluator.

The extension provides:

1. A single-support body-acceleration linear-static family that raises
   solver-qualified item coverage to 164/300.
2. Evaluate explicit geometric-scale laws on synthetic, topology-preserving
   STEP variants without treating those variants as new independent dataset
   items.
3. Matched reference/candidate response scoring under fixed acceleration for
   regularized p95 stress, normalized displacement, and dimensionless
   body-force stress concentration.

All scenarios remain benchmark-assigned and set `source_intent_claimed: false`.
The extension is calibrated and accepted for the `extended_v1` benchmark
profile. It remains a benchmark-assigned probe, not a source-author service
load or safety certification.
