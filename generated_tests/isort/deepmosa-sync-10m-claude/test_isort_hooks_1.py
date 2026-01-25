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
    assert var_0 == 0

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = True
    var_1 = 'J.47m5Ge'
    var_2 = "Qd1)jx'_yf?ECQ"
    var_3 = 'a4S1;`jrF8#m0/_rQAu'
    var_4 = [var_1, var_2, var_1, var_3]
    module_0.git_hook(var_0, directories=var_4)

def test_case_3():
    var_0 = 'echo'
    var_1 = 'single'
    var_2 = [var_0, var_1]
    var_3 = module_0.get_lines(var_2)
    var_4 = bool(var_3 == ['single line'])