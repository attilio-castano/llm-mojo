"""Fast selection is exactly the measured M4 Pro lookup; everything else is baseline.

The measured cells come from the retained runtime study, so a change to the
lookup cannot silently drift from the evidence that justified it.
"""
from std.testing import TestSuite, assert_equal, assert_raises
from llm_mojo.models.qwen2.plan import ExecutionPlan, MEASURED_DEVICE, configured_plan, execution_plan

comptime M4_PRO = MEASURED_DEVICE


def _fields(plan: ExecutionPlan) -> SIMD[DType.int64, 4]:
    return SIMD[DType.int64, 4](Int64(plan.configuration), Int64(1 if plan.gpu_argmax else 0),
                                Int64(1 if plan.swap_buffers else 0), Int64(1 if plan.fuse_residual_norm else 0))


def _plan(mode: String, rows: Int, total: Int, device: String) raises -> SIMD[DType.int64, 4]:
    """(configuration, GPU argmax, buffer swap, residual/RMSNorm fusion) for one call."""
    return _fields(execution_plan(mode, rows, total, device))


def _cells() raises -> List[SIMD[DType.int64, 4]]:
    var cells = List[SIMD[DType.int64, 4]]()
    var lines = open("studies/model_generation/runtime-selection.csv", "r").read().splitlines()
    assert_equal(String(lines[0]), "rows,total,configuration")
    for i in range(1, len(lines)):
        var fields = String(lines[i]).split(",")
        cells.append(SIMD[DType.int64, 4](Int64(Int(String(fields[0]))), Int64(Int(String(fields[1]))),
                                          Int64(Int(String(fields[2]))), 0))
    return cells^


def test_measured_prefill_cells_follow_the_runtime_study() raises:
    var cells = _cells()
    assert_equal(len(cells), 11)
    for cell in cells:
        var expected = SIMD[DType.int64, 4](cell[2], 0, 0, 0)
        assert_equal(_plan("fast", Int(cell[0]), Int(cell[1]), M4_PRO), expected)


def test_single_rows_use_the_fused_decode_route() raises:
    for total in [1, 64, 65, 1024, 3968, 4096]:
        assert_equal(_plan("fast", 1, total, M4_PRO), SIMD[DType.int64, 4](26, 1, 1, 1))


def test_unmeasured_cells_and_devices_use_the_baseline() raises:
    var neighbours: List[Int] = [16, 255, 16, 257, 14, 256, 63, 4096, 64, 1023, 256, 256, 4096, 4096]
    for i in range(0, len(neighbours), 2):
        assert_equal(_plan("fast", neighbours[i], neighbours[i + 1], M4_PRO), SIMD[DType.int64, 4](0))
    var cells = _cells()
    for device in ["Apple M1", "", "Apple M4 Pro (x)"]:
        for cell in cells:
            assert_equal(_plan("fast", Int(cell[0]), Int(cell[1]), device), SIMD[DType.int64, 4](0))
        assert_equal(_plan("fast", 1, 64, device), SIMD[DType.int64, 4](0))


def test_research_modes_are_fixed_and_dimensions_are_checked() raises:
    for rows in [1, 16, 256]:
        assert_equal(_plan("baseline", rows, 1024, M4_PRO), SIMD[DType.int64, 4](0))
        assert_equal(_plan("consistent", rows, 1024, M4_PRO), SIMD[DType.int64, 4](20, 0, 0, 0))
    with assert_raises():
        _ = _plan("fast", 0, 1, M4_PRO)
    with assert_raises():
        _ = _plan("fast", 2, 1, M4_PRO)
    with assert_raises():
        _ = _plan("fast", 1, 4097, M4_PRO)


def test_retired_mode_names_and_unmeasured_compositions_are_rejected() raises:
    for name in ["auto", "all-three", "projection-0", "combined", "gpu-argmax", "candidate", "20", ""]:
        with assert_raises():
            _ = execution_plan(name, 1, 64, M4_PRO)
    # Configuration 26 carries all three decode features on exactly one row; nothing else carries any.
    ExecutionPlan(26, True, True, True).validate(1)
    for bad in [ExecutionPlan(26, False, True, True), ExecutionPlan(26, True, False, True),
                ExecutionPlan(26, True, True, False), ExecutionPlan(0, True, False, False),
                ExecutionPlan(0, False, True, False), ExecutionPlan(3, False, False, True)]:
        with assert_raises():
            bad.validate(1)
    with assert_raises():
        ExecutionPlan(26, True, True, True).validate(2)
    assert_equal(_fields(configured_plan(26, 1, 64)), SIMD[DType.int64, 4](26, 1, 1, 1))
    assert_equal(_fields(configured_plan(21, 16, 64)), SIMD[DType.int64, 4](21, 0, 0, 0))
    with assert_raises():
        _ = configured_plan(26, 2, 64)
    with assert_raises():
        _ = configured_plan(999, 1, 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
