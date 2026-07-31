# Representative Dataset Subsets

## CADEngBench-P sample10

The P subset is derived from the frozen evaluator development slice rather than
a fresh unstratified random sample. It has six BenchCAD and four Fusion-derived
items across easy, medium, and hard tiers and covers both supported source
representations, zero-to-CAD, functional edits, parameter perturbations, and
single/multi-solid geometry. Every selected object is solver-qualified for L3.

Each object stores its L3 annotation context, annotation provenance and
validation, solver-qualified physics card, source-qualification records, and
calibrated response contract directly under `items/<item_id>/l3/`. There is no
separate L3 sidecar or fixture dataset in the supplement.

The included `slice_plan.json` and `development_slice_manifest.json` record the
selection and control bindings. Positive and semantic-mutant submissions allow
provider-free L0-L2 calibration checks.

## CADEngBench-A sample7

The A subset contains one development item for each of Ball, Cylindrical,
Pin-Slot, Planar, Revolute, Rigid, and Slider joint families. It includes public
geometry, views, candidate registries, exact A1/A2 prompts, and development
labels. No held-out Core-150 labels are included.

## Why not purely random?

AAAI asks for a representative subset when the full data do not fit. A small
unstratified random draw could omit a source, difficulty tier, or joint family.
The existing audited development slices provide broader reviewer coverage while
remaining compact and clearly separated from held-out evaluation data.
