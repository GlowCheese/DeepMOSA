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

def test_case_2():
    pass

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = True
    module_0.git_hook(directories=var_0)

def test_case_4():
    var_0 = 'echo'
    var_1 = 'line1\nline2\nline3'
    var_2 = [var_0, var_1]
    var_3 = module_0.get_lines(var_2)