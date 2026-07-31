# CADEngBench-A Pair-Level Metric Specification v1.2

> **Historical protocol notice (16 July 2026):** This specification treated every consolidated source record as a joint instance whose multiplicity should be reproduced. JoinABLe's Joint Consolidation procedure shows that records in a `joint_set` are alternative known positives observed for an identical part pair, not necessarily simultaneous constraints. Keep v1.2 for immutable regression evidence only. New model evaluation uses `CADEngBench_A_Known_Joint_Hypothesis_Metric_Specification_v2.md`; v1 multiplicity and one-to-one instance scores must not be reported as benchmark-quality model metrics.

**Artifact class:** Versioned source specification

**Specification ID:** `cadengbench_a_pair_metrics_v1_2`
**Ground truth:** `ground_truth/master.json`
**Evaluation unit:** one Fusion `joint_set` containing two CAD bodies and one or more joint instances
**Split:** `cadengbench_a_grouped_split_v1`

## 1. Scope

This specification evaluates whether a system predicts the mechanical relationship between the two CAD bodies supplied in a pair-level CADEngBench-A item. It evaluates output validity, joint-instance recovery, Fusion joint type, coarse degrees of freedom, joint-axis geometry, and selected BRep entities.

The evaluator compares predictions directly with CAD-native normalized ground truth. URDF and PyBullet are not sources of truth; their results remain separate simulation-compatibility diagnostics.

## 2. Required Prediction Fields

Each prediction file must contain:

```json
{
  "assembly_id": "joint_set_00000",
  "predicted_joints": [
    {
      "body_a": "body_id_a",
      "body_b": "body_id_b",
      "joint_type": "RevoluteJointType",
      "axis": {
        "origin": [0.0, 0.0, 0.0],
        "direction": [0.0, 0.0, 1.0]
      },
      "selected_entities": {
        "body_a": {"type": "BRepFace", "index": 3},
        "body_b": {"type": "BRepEdge", "index": 7}
      }
    }
  ]
}
```

`selected_entities` is optional in v1. When omitted, entity prediction rate and entity accuracy receive zero credit for the corresponding sides, but the output remains L0-valid.

## 3. L0 Output Validity

An item is L0-valid when:

- the prediction is parseable JSON;
- `predicted_joints` is a list;
- the assembly ID is absent or matches the evaluated item;
- every predicted body ID belongs to the item;
- every joint type is in the seven-type Fusion vocabulary;
- every supplied axis has a finite three-dimensional origin and a finite nonzero direction;
- an omitted axis carries a recognized unresolved status rather than fabricated coordinates;
- every supplied entity reference contains `type` (`BRepFace` or `BRepEdge`) and an integer `index`.

An L0-invalid item receives zero diagnostic composite score. It cannot contribute true-positive matches to official micro metrics.

Primary validity metric:

```text
L0 validity rate = valid items / evaluated items
```

## 4. Multi-Joint Matching

Ground-truth and predicted joints are not matched by list position. The evaluator computes a one-to-one best assignment using:

- unordered body-pair identity;
- exact joint type;
- coarse DOF compatibility;
- sign-invariant axis alignment;
- axis-line distance;
- exact or equivalent entity references.

For up to eight truth and eight predicted joints, the assignment is solved exactly using dynamic programming. Larger unusually dense pairs use a deterministic greedy fallback to keep runtime bounded. The report should disclose how many evaluated items use the fallback when this becomes material.

## 5. Typed Joint-Instance Recovery

A joint instance is recovered when its assigned prediction uses both the correct unordered body pair and the correct Fusion joint type. Invalid items produce no true positives.

Primary metrics are globally micro-aggregated:

```text
precision = matched joint instances / predicted joint instances
recall    = matched joint instances / ground-truth joint instances
F1        = harmonic mean of precision and recall
```

`joint_count_exact_rate` is reported as a secondary item-level metric.

Because pair-level items already supply the two candidate bodies, the evaluator separately reports **known-pair multiplicity precision, recall, and F1**. These metrics require the correct unordered body pair but do not require the correct joint type. They measure whether a method recovered the number of repeated constraints on the supplied pair; they are diagnostics, not substitutes for typed joint-instance F1.

## 6. Joint-Type Metrics

The recognized labels are:

- `RigidJointType`
- `RevoluteJointType`
- `SliderJointType`
- `CylindricalJointType`
- `PlanarJointType`
- `BallJointType`
- `PinSlotJointType`

Primary metrics:

- joint-type accuracy conditional on known-pair assignments;
- per-type precision, recall, and F1;
- macro F1 across types present in truth or predictions.

The evaluator also reports the number of truth, predicted, and true-positive instances for every type.

## 7. Coarse DOF Compatibility

DOF compatibility is derived from the predicted and true joint types:

| Fusion joint type | Translational DOF | Rotational DOF |
| --- | ---: | ---: |
| Rigid | 0 | 0 |
| Revolute | 0 | 1 |
| Slider | 1 | 0 |
| Cylindrical | 1 | 1 |
| Planar | 2 | 1 |
| Ball | 0 | 3 |
| Pin-slot | 1 | 1 |

The primary DOF metric is compatibility accuracy on known-pair assignments. This is intentionally coarse: cylindrical and pin-slot joints receive the same DOF signature even though their geometric constraint families differ. Joint-type metrics retain that distinction.

## 8. Axis Geometry

Primary axis metrics apply only when:

- body pair and joint type are correct; and
- the true type is Revolute, Slider, Cylindrical, Planar, or PinSlot.

Rigid and Ball joints are excluded because a single axis line is not their defining kinematic quantity.

Angular error is sign-invariant:

```text
angle = acos(abs(dot(unit(predicted_direction), unit(true_direction))))
```

This treats directions `d` and `-d` as the same geometric axis.

Axis distance is the shortest distance between the predicted and true infinite lines. The evaluator reports:

- mean and median angular error in degrees;
- mean and median raw line distance in source CAD units;
- mean and median normalized line distance.

Normalized distance divides raw line distance by the largest body bounding-box diagonal in the pair. This is the preferred scale-comparable distance metric.

If a method produces no type-correct axis-eligible predictions, its axis metric is `n/a`, not zero.

The evaluator separately reports axis-required count, axis-predicted count, and axis prediction rate. Track B candidate selection operates on isolated bodies in separate local coordinate frames. Autodesk documents that the assembled transforms live in each hidden joint record; therefore the Track B adapter reports `unresolved_no_shared_assembly_frame` and receives `n/a` for axis error instead of using ground truth to construct an axis. A future pose-aware track may score shared-frame axes when it supplies a non-leaking coordinate-frame contract.

Format reference: <https://github.com/AutodeskAILab/Fusion360GalleryDataset/blob/master/docs/assembly_joint.md#joint-geometry>

## 9. BRep Entity References

Each joint has a primary selected entity for both body sides. Most records also provide geometrically equivalent alternatives.

Entity identity is evaluated within its body side using:

```text
(entity type, entity index)
```

Primary entity metrics:

- **prediction rate:** fraction of truth entity sides for which the method supplied a valid entity reference;
- **exact accuracy:** prediction matches the primary Fusion reference;
- **equivalence-aware accuracy:** prediction matches the primary reference or one of its recorded equivalent references.

Equivalence-aware accuracy should be the main entity-selection metric because the same valid mate geometry may be represented by an equivalent cylindrical face or circular edge.

## 10. Official Primary Metrics

The v1 headline table should contain:

1. L0 validity rate
2. Typed joint-instance precision, recall, and F1 (micro)
3. Known-pair multiplicity F1 (diagnostic context)
4. Joint-type accuracy on known-pair assignments
5. Joint-type macro F1
6. DOF compatibility accuracy
7. Primary-axis prediction rate
8. Primary-axis median angular error
9. Primary-axis median normalized distance error
10. Entity prediction rate
11. Entity equivalence-aware accuracy

Raw means, exact entity accuracy, joint-count exact rate, URDF lossiness slices, and the experimental composite score are secondary diagnostics.

No single composite number is an official v1 benchmark score. A method should not be able to hide poor type or entity reasoning behind one weighted average.

## 11. Train/Validation/Test Protocol

The official v1 split groups pair records by reconstructed connected-component graph before assigning them to train, validation, or test. This prevents the same body or surrounding assembly context from crossing partitions.

Current counts:

| split | Pair items | Graph groups | Unique bodies |
| --- | ---: | ---: | ---: |
| Train | 705 | 490 | 1,294 |
| Validation | 151 | 111 | 278 |
| Test | 144 | 102 | 262 |

All seven joint types occur in every split. Body overlap and graph-group overlap are both zero.

Models may use training labels to learn parameters and validation labels for model selection. Test labels must be used only for final evaluation. Any baseline using class frequencies must estimate those frequencies from the training split only.

## 12. Required Reporting Slices

Results should be reported:

- overall on validation and test;
- per joint type;
- single-joint versus multi-joint items;
- by URDF lossiness label;
- by selection/complexity bucket;
- with entity metrics shown separately from type and axis metrics.

Simulation pass rate remains a separate supporting result and must not replace CAD-native correctness.

## 13. Known v1 Limitations

- The pair task provides the two candidate bodies, so known-pair multiplicity is not evidence of body-pair discovery and must not be presented as typed joint correctness.
- Coarse DOF compatibility does not distinguish all geometric constraint families.
- Entity scoring uses normalized body-local BRep indices and recorded equivalences; it does not yet score semantic face regions.
- Mate-frame translation and orientation are preserved in ground truth but are not yet official v1 metrics.
- Dense items above eight-by-eight matching use a deterministic greedy assignment fallback.
- The graph split is based on reconstructed connectivity, not official Fusion hierarchy.

These limitations are explicit so v1 remains reproducible and does not overstate what the current data supports.
