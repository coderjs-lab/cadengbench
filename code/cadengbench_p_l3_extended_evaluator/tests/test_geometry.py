from cadengbench_p_l3_extended.geometry import extension_candidate


def _geometry(*, solids: int, regions: int) -> dict:
    return {
        "solid_count": solids,
        "qualified_single_support_regions": [
            {"region_id": f"axis:x:{index}"} for index in range(regions)
        ],
    }


def test_extension_candidate_requires_new_single_solid_with_support() -> None:
    assert extension_candidate(
        _geometry(solids=1, regions=1), existing_solver_qualified=False
    ) == (True, "single_support_body_acceleration_candidate")
    assert extension_candidate(
        _geometry(solids=1, regions=1), existing_solver_qualified=True
    ) == (False, "already_solver_qualified_v1")
    assert extension_candidate(
        _geometry(solids=2, regions=1), existing_solver_qualified=False
    ) == (False, "solid_count_not_one")
    assert extension_candidate(
        _geometry(solids=1, regions=0), existing_solver_qualified=False
    ) == (False, "no_qualified_single_support_region")
