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
    var_3 = 'Decorator'
    var_4 = ()
    var_5 = 'lineno'
    var_6 = 10
    var_7 = {var_5: var_6}
    var_8 = type(var_3, var_4, var_7)
    var_9 = var_8()
    var_10 = [var_9]
    var_11 = {var_2: var_10}
    var_12 = type(var_0, var_1, var_11)
    var_13 = module_0.get_first_line_number(var_12)
    assert var_13 == 10