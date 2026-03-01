# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.hooks as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = True
    module_0.git_hook(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    module_0.git_hook()

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = True
    var_1 = 'J.47m5Ge'
    var_2 = "Qd1)jx'_yf?ECQ"
    var_3 = 'a4S1;`jrF8#m0/_rQAu'
    var_4 = [var_1, var_2, var_1, var_3]
    module_0.git_hook(var_0, directories=var_4)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'echo'
    var_1 = '-e'
    var_2 = 'line1\nline2\nline3'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.get_lines(var_3)
    var_5 = '  line1  \n\tline2\t\n  line3  '
    var_6 = [var_0, var_1, var_5]
    var_7 = module_0.get_lines(var_6)
    var_8 = '-n'
    var_9 = [var_0, var_8]
    var_10 = module_0.get_lines(var_9)
    var_11 = module_0.get_lines(var_3)
    var_12 = True
    module_0.git_hook(var_12)