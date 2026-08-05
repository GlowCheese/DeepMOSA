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

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = 'Node'
    var_1 = ()
    var_2 = 'decorator_list'
    var_3 = 'lineno'
    var_4 = 15
    var_5 = {var_2: var_3, var_3: var_4}
    var_6 = type(var_0, var_1, var_5)
    module_0.get_first_line_number(var_6)
    assert var_7 == 15