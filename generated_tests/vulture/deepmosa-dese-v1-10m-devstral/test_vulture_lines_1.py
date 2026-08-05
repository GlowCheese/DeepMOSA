# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import vulture.lines as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.get_first_line_number(var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.get_last_line_number(var_0)