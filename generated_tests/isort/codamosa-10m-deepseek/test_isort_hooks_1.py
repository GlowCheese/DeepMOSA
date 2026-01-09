# Check out: https://github.com/GlowCheese/deepmosa
import isort.hooks as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = True
    module_0.git_hook(lazy=var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    module_0.git_hook()

def test_case_2():
    pass

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = False
    var_1 = False
    var_2 = False
    var_3 = '#F5\x0ciCuxYlUe'
    var_4 = '\nxE-Q'
    var_5 = [var_3, var_4]
    module_0.git_hook(var_0, var_1, var_2, var_3, var_5)

def test_case_4():
    var_0 = 'echo'
    var_1 = 'Hello\nWorld'
    var_2 = [var_0, var_1]
    var_3 = module_0.get_lines(var_2)
    var_4 = module_0.get_lines(var_2)
    var_5 = 'All tests passed!'
    var_6 = print(var_5)