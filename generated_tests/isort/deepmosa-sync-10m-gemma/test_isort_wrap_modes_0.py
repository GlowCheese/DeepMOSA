# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.wrap_modes as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.from_string(var_0)

def test_case_1():
    pass

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.formatter_from_string(var_0)

def test_case_3():
    var_0 = 'x = 1'
    var_1 = 'import os'
    var_2 = [var_1]
    var_3 = ' '
    var_4 = '    '
    var_5 = 80
    var_6 = '# comment'
    var_7 = [var_6]
    var_8 = '\n'
    var_9 = '#'
    var_10 = True
    var_11 = False
    var_12 = module_0._wrap_mode_interface(var_0, var_2, var_3, var_4, var_5, var_7, var_8, var_9, var_10, var_11)
    assert var_12 == ''

def test_case_4():
    var_0 = 'hello '
    var_1 = module_0._hanging_indent_end_line(var_0)
    assert var_1 == 'hello \\'

def test_case_5():
    var_0 = 'hello'
    var_1 = module_0._hanging_indent_end_line(var_0)
    assert var_1 == 'hello \\'