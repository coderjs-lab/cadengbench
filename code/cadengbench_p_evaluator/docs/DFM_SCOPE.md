# Initial DFM Scope

DFM is an internal L1 component, not a fourth public layer. This implementation
contains exactly two evidence-gated families.

## Minimum wall thickness

The evaluator combines supported analytic candidates with a deterministic sampled
estimate and reports the minimum observed value. Pass/fail is permitted only when
the case binds all of the following:

1. declared applicability and a public manufacturing condition;
2. a finite positive minimum thickness;
3. threshold evidence;
4. `scoreable: true`; and
5. a certified global-minimum-thickness extractor for the item's geometry class.

Source-explicit nominal wall values and parallel-face/coaxial-cylinder proxies do
not by themselves certify a global minimum and remain `not_scoreable` diagnostics.

## Minimum hole geometry

For unambiguous internal near-complete cylindrical holes, the evaluator measures
minimum diameter and maximum depth-to-diameter ratio. Pass/fail additionally needs
finite positive limits, evidence, and a certified recognizer. If the contract
declares a required hole count, a certified recognizer can fail missing holes; this
prevents a model from evading DFM by deleting every hole. If no required count is
frozen and no supported hole is recognized, the result is `not_scoreable`, not a pass.

Blind, threaded, intersecting, non-cylindrical, draft, undercut, accessibility,
internal-radius, additive, CAM, and toolpath checks are deliberately out of scope.
They must not be approximated under these two family names.
