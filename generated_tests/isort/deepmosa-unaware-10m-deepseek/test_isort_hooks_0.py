# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.hooks as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    var_1 = True
    module_0.git_hook(modify=var_0, lazy=var_1, directories=var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    module_0.git_hook()

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = ' /w9c*76GuYJ:'
    module_0.git_hook(directories=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'echo'
    var_1 = 'test'
    var_2 = [var_0, var_1]
    var_3 = module_0.get_lines(var_2)
    var_4 = '&MFu+'
    var_5 = 'test'
    var_6 = [var_4, var_5]
    module_0.get_lines(var_6)