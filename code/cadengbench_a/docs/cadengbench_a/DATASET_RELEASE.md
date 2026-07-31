# CADEngBench-A Dataset Release

## Core-150

| Property | Value |
| --- | ---: |
| Total items | 150 |
| Development items | 30 |
| Held-out test items | 120 |
| Public input package | approximately 245 MB |
| Evaluator-only label files | 150 |
| Joint families | 7 |
| Maximum estimated input size | 98,505 tokens |
| Items above 128k estimated input | 0 |
| Maximum images in one request | 31 |

Each public item contains two isolated body representations, STEP and OBJ geometry,
standardized clean views, candidate atlases, a stable B-Rep candidate registry,
public geometry summaries, provenance, and a task contract. Joint labels, source
entities, frames, and motion semantics are held separately for evaluation.

The checked-in manifest
`dataset/cadengbench_a/releases/core150_public_v1_manifest.json` records every
public path, byte count, SHA-256, partition, and evaluator-label hash. The full
binary package is distributed as a release artifact rather than ordinary Git
content.

## Compact submission sample

`submission/cadengbench_a_sample7_v2/` is a 7.2 MB development-only package with
one representative for each family:

| Joint family | Assembly ID |
| --- | --- |
| Ball | `joint_set_17859` |
| Cylindrical | `joint_set_02236` |
| Pin-slot | `joint_set_15485` |
| Planar | `joint_set_01993` |
| Revolute | `joint_set_09025` |
| Rigid | `joint_set_02495` |
| Slider | `joint_set_18632` |

The sample includes development labels for reviewer reproduction and contains no
held-out test label.
