# CADEngBench-A Pair Track B Prompt Card v3

- **Task:** Rank up to three plausible joint hypotheses for two supplied CAD bodies.
- **Status:** Two-reviewer input-package audit passed; selected as the benchmark-contract prompt. Final model benchmarking remains deferred.
- **Change from v2:** Candidate face groups are color-highlighted in the atlas, edge candidates use X key-point markers, and the prompt specifies a geometry-first decision procedure.
- **Evidence basis:** Four of six Dev20 v2 top-3 misses were entity-selection misses; miss items had a median 4,871 candidate pairs versus 766 for recovered items.
- **Dataset semantics:** A `joint_set` consolidates alternative known positives, not simultaneous constraints or exhaustive negatives.
- **Output:** Up to three ranked joint-family and body-A/body-B candidate-ID hypotheses, or bounded abstention.
- **Forbidden evidence:** Source joint type/count/axis/entity labels, assembled preview, URDF, PyBullet, validation status, evaluator output, and test labels.
- **Leakage boundary:** Face coloring is derived only from OBJ `g face N` groups and public candidate indices. No known joint is used to choose a color or candidate.
- **Scoring:** Same positive-and-unlabeled top-1/top-3 known-positive metrics as v2, enabling a controlled representation/prompt ablation.
- **Immutability:** The completed v2 run and package remain unchanged.
