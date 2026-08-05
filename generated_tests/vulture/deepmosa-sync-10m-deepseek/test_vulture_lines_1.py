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

def test_case_2():
    var_0 = 'Node'
    var_1 = ()
    var_2 = 'decorator_list'
    var_3 = 'lineno'
    var_4 = 'Decorator'
    var_5 = ()
    var_6 = 5
    var_7 = {var_3: var_6}
    var_8 = type(var_4, var_5, var_7)
    var_9 = var_8()
    var_10 = [var_9]
    var_11 = 10
    var_12 = {var_2: var_10, var_3: var_11}
    var_13 = type(var_0, var_1, var_12)
    var_14 = var_13()
    var_15 = module_0.get_first_line_number(var_14)
    assert var_15 == 5