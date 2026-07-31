# CADEngBench-A Known-Joint Hypothesis Metric Specification v2

**Artifact class:** Versioned source specification  
**Specification ID:** `cadengbench_a_known_joint_hypothesis_metrics_v2`  
**Evaluation unit:** one consolidated Fusion `joint_set` for a geometrically identical pair of CAD parts  
**Task:** rank plausible joint-type and BRep-entity hypotheses  
**Status:** Implemented and selected for the construction contract; final model benchmarking deferred

## 1. Dataset Semantics

The Fusion Gallery Assembly-Joint data used by JoinABLe consolidates joints observed between geometrically identical part pairs across source assemblies. A `joint_set` may therefore contain several **alternative known-positive joint labels**. Its records are not assumed to be simultaneous constraints in one mechanism, and the labels are not an exhaustive list of all mechanically plausible joints.

This is a positive-and-unlabeled task:

- a match to a recorded label is a known-positive hit;
- a non-match is not proof that the prediction is mechanically impossible;
- raw source-record count is provenance, not a model prediction target;
- exact duplicate source records must not inflate task complexity or require duplicate outputs.

This interpretation follows JoinABLe's Joint Consolidation procedure and its loader, which merges all known joint entity pairs for a consolidated part pair into one positive label matrix.

## 2. Public Input

The model receives only:

- two isolated body IDs;
- standardized body-local renders;
- deterministic public geometry facts;
- complete deterministic BRep candidate registries;
- the seven Fusion joint-family definitions;
- the versioned output schema.

The public package must not contain source joint count, type, selected entities, axis, assembled preview, URDF status, PyBullet results, validation outcome, ground-truth paths, or test labels.

## 3. Prediction Contract

The model returns zero to three ranked alternative hypotheses:

```json
{
  "assembly_id": "joint_set_00000",
  "ranked_hypotheses": [
    {
      "rank": 1,
      "joint_type": "RevoluteJointType",
      "body_a_candidate_id": "body_a_face_0003",
      "body_b_candidate_id": "body_b_edge_0007",
      "confidence": 0.72
    }
  ],
  "abstention": null
}
```

Hypotheses are alternatives in descending plausibility. They are not a list of simultaneous constraints and are not an estimate of source-record multiplicity.

## 4. Source Accounting

Every raw record and source index is retained. The evaluator additionally computes an exact normalized record key after removing internal parser-only IDs and derived URDF fields. Records with the same key form one exact source-variant group.

Per item, report:

- `source_record_count`;
- `exact_unique_source_variant_count`;
- `exact_duplicate_extra_count`;
- known joint-family set;
- union of primary and source-recorded equivalent BRep candidates for each body side.

Exact deduplication is deliberately conservative. Geometrically redundant but non-identical source records remain separate known variants and may be analyzed later.

## 5. Output Validity

An item is valid when:

- the output is schema-valid JSON;
- the assembly ID matches;
- ranks are unique, consecutive, and ordered from one;
- every joint type is in the seven-type Fusion vocabulary;
- every candidate ID exists and belongs to the declared body side;
- no two hypotheses are identical;
- confidence is finite and in `[0, 1]`;
- an abstention is used only with an empty hypothesis list.

Strict first-pass validity and syntax-only recovered validity are reported separately. Semantic repair is prohibited.

## 6. Known-Positive Matching

For each source joint, the evaluator forms the Cartesian product of accepted candidates on body A and body B. Accepted candidates include the primary selected BRep entity and all source-recorded equivalent entities that resolve in the public registry.

A prediction receives:

- **known type hit** when its joint family occurs in any known-positive source record;
- **known-positive entity hit** when its candidate pair matches a known-positive entity pair, independent of type;
- **known-positive typed-entity hit** when both the joint family and candidate pair match the same known-positive source record.

Candidate matching is role-aware. Body A and body B IDs are not swapped because the public package fixes those roles.

## 7. Headline Metrics

Report micro means and bootstrap confidence intervals over graph-grouped items for:

1. strict L0 validity rate;
2. top-1 known-positive entity hit rate;
3. top-3 known-positive entity hit rate;
4. top-1 known-positive typed-entity hit rate;
5. top-3 known-positive typed-entity hit rate;
6. typed-entity mean reciprocal rank;
7. abstention rate;
8. hallucinated candidate-ID rate.

The hallucinated candidate-ID rate is a micro ratio over attempted candidate-ID
references: total invalid or wrong-side candidate-ID references divided by total
candidate-ID references. It is not an unweighted mean of per-item rates.

Secondary diagnostics:

- top-1 and top-3 known joint-family hit;
- results per joint family;
- results by exact unique known-variant count;
- candidate count and visual-atlas burden;
- latency, token usage, and cost;
- exact duplicate prevalence as a dataset diagnostic.

Exact source-variant deduplication ignores record IDs, source indices, and
designer display names. Those fields do not change joint geometry or motion.

No single weighted composite is an official score.

## 8. What Is Not Scored

The v2 pair task does not score:

- reproduction of raw source-record count;
- simultaneous multi-constraint mechanism recovery;
- chronological assembly order;
- full assembly-graph reconstruction;
- world-frame axis error when only isolated body-local frames are public;
- URDF tree conversion or PyBullet motion as ground truth;
- mechanical impossibility of an unlabelled hypothesis.

Those require different evidence or a separate task contract.

## 9. Selection Policy

Dev and test cohorts must be selected using:

- exact unique known-variant count for complexity;
- joint-family coverage;
- body geometry/candidate burden;
- graph-grouped split membership.

Raw source-record count and exact duplicate extras are provenance-only. URDF lossiness is not a selection quota for the CAD-native hypothesis task. The frozen test split remains untouched until prompt, evaluator, and human-audit policy are frozen on validation.

## 10. Version Boundary

Pair Metric v1.2 remains an immutable historical implementation and remains useful for oracle regression tests over normalized records. Its multiplicity and one-to-one joint-instance metrics must not be presented as valid model-quality metrics for consolidated `joint_set` prediction.

Smoke3 v1 remains preserved as transport/evaluator evidence. Model-quality interpretation moves to Prompt Track B v2 and this metric specification.

## 11. References

1. Willis et al., [JoinABLe: Learning Bottom-up Assembly of Parametric CAD Joints](https://openaccess.thecvf.com/content/CVPR2022/papers/Willis_JoinABLe_Learning_Bottom-Up_Assembly_of_Parametric_CAD_Joints_CVPR_2022_paper.pdf), CVPR 2022.
2. Autodesk AI Lab, [JoinABLe dataset loader](https://github.com/AutodeskAILab/JoinABLe/blob/main/datasets/joint_graph_dataset.py).
3. Autodesk AI Lab, [Fusion 360 Gallery Assembly Dataset documentation](https://github.com/AutodeskAILab/Fusion360GalleryDataset/blob/master/docs/assembly.md).
