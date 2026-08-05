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
    var_2 = 'lineno'
    var_3 = 'decorator_list'
    var_4 = 10
    var_5 = [var_0]
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = type(var_0, var_1, var_6)
    module_0.get_first_line_number(var_7)