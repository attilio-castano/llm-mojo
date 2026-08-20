from llm_mojo import PROJECT_NAME
from std.testing import TestSuite, assert_equal


def test_package_import() raises:
    assert_equal(PROJECT_NAME, "llm-mojo")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
