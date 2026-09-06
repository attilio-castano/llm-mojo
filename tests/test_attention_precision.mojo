"""Primary FP32 attention accuracy gate plus named BF16 comparisons."""
from std.python import Python
from std.sys import get_defined_int
from std.testing import TestSuite
from test_attention_sublayer import _case
from test_attention_sublayer_operations import _operations


def _run_cases(composed: Bool) raises:
    var json = Python.import_module("json")
    var pathlib = Python.import_module("pathlib")
    var selected = get_defined_int["PRECISION_CASE", default=-1]()
    var checkpoint = get_defined_int["PRECISION_CHECKPOINT", default=0]()
    var filename = "precision_manifest.json"
    if selected >= 0:
        filename = "precision_case_" + String(selected) + ".json"
    if checkpoint:
        filename = "checkpoint_manifest.json"
    var manifest = json.loads(pathlib.Path(
        "build/oracle_data/attention_sublayer/" + filename
    ).read_text())
    var specs = manifest["cases"]
    var offset = 17 if checkpoint else 0
    var failed_cases = 0
    for i in range(Int(py=specs.__len__())):
        var case_id = i + offset
        if selected >= 0 and selected != case_id:
            continue
        var nq = Int(py=specs[i][0])
        var nk = Int(py=specs[i][1])
        var d = Int(py=specs[i][2])
        var t = Int(py=specs[i][3])
        print("precision case", case_id, "composed", composed)
        if not composed:
            try:
                _operations(case_id, nq, nk, d, t, True)
            except:
                failed_cases += 1
            continue
        for route in range(7):
            if nq != 14 and route != 0 and route != 3:
                continue
            for chunked in range(2):
                try:
                    _case(case_id, nq, nk, d, t, route, Bool(chunked), "fp32", route >= 3)
                except:
                    failed_cases += 1
        # Same original X, fixed FP32 attention policy and unchanged gates;
        # only Wo's work ownership changes. Includes tiny/ragged shapes.
        for chunked in range(2):
            try:
                _case(case_id, nq, nk, d, t, 3, Bool(chunked), "fp32", True, True)
            except:
                failed_cases += 1
            if nq == 14:
                try:
                    _case(case_id, nq, nk, d, t, 6, Bool(chunked), "fp32", True, True)
                except:
                    failed_cases += 1
    if failed_cases:
        raise Error("FP32 attention failed its declared accuracy or exact cache gates")


def test_precision_operations_against_declared_upstream() raises:
    _run_cases(False)


def test_precision_composition_against_declared_upstream() raises:
    _run_cases(True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
