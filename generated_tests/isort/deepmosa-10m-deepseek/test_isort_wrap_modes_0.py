# Check out: https://github.com/GlowCheese/deepmosa
import isort.wrap_modes as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = '^J"|f\rROJ!$=c'
    module_0.from_string(var_0)

def test_case_1():
    pass

def test_case_2():
    var_0 = 'n-Q>V\\{%&'
    var_1 = module_0.formatter_from_string(var_0)

def test_case_3():
    var_0 = 'INVALID'
    var_1 = False
    var_2 = None
    var_3 = module_0._wrap_mode_interface(var_0, var_0, var_0, var_2, var_2, var_2, var_0, var_0, var_1, var_2)
    assert var_3 == ''

def test_case_4():
    var_0 = 'test   '
    var_1 = module_0._hanging_indent_end_line(var_0)
    assert var_1 == 'test   \\'

def test_case_5():
    var_0 = 'test'
    var_1 = module_0._hanging_indent_end_line(var_0)
    assert var_1 == 'test \\'