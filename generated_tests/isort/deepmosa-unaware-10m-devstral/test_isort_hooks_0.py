# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.hooks as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    module_0.git_hook()

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = 'echo'
    var_1 = 'line1\nline2\nline3'
    var_2 = [var_0, var_1]
    var_3 = module_0.get_lines(var_2)
    var_4 = None
    var_5 = True
    module_0.git_hook(modify=var_4, lazy=var_5, settings_file=var_4)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = '[BbZ]'
    var_1 = 'xrr[.*mc?\x0b4C{e,ll'
    var_2 = [var_0, var_1, var_0]
    module_0.git_hook(directories=var_2)

def test_case_3():
    var_0 = 'echo'
    var_1 = '-e'
    var_2 = [var_0, var_1, var_0]
    var_3 = module_0.get_lines(var_2)
    var_4 = 'single_line'
    var_5 = [var_0, var_4]
    var_6 = module_0.get_lines(var_5)
    var_7 = '-n'
    var_8 = [var_0, var_7]
    var_9 = module_0.get_lines(var_8)
    var_10 = '  line1  \n  line2  \n  line3  '
    var_11 = [var_0, var_1, var_10]
    var_12 = module_0.get_lines(var_11)