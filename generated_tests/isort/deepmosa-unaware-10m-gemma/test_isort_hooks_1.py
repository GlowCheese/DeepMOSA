# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.hooks as module_0
import ast as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = True
    var_1 = None
    module_0.git_hook(var_0, var_0, var_0, directories=var_1)

@pytest.mark.xfail(strict=True)
def test_case_1():
    module_0.git_hook()

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = module_1._Precedence.TUPLE
    var_1 = True
    module_0.git_hook(var_1, var_0, directories=var_0)

def test_case_3():
    var_0 = 'test'
    var_1 = [var_0, var_0]
    var_2 = module_0.get_lines(var_1)