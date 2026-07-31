"""CADEngBench-P L3 annotation and finite-element evaluation."""

from .annotation import (
    compile_physics_cards,
    prepare_annotation_corpus,
    run_annotation_corpus,
)

__all__ = [
    "compile_physics_cards",
    "prepare_annotation_corpus",
    "run_annotation_corpus",
]
