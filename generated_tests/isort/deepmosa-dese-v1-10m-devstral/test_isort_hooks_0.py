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

def test_case_2():
    pass

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'hHCgcWb\r:f'
    var_1 = '+*l\tMr\x0cD_-'
    var_2 = [var_0, var_0, var_1, var_1]
    module_0.git_hook(directories=var_2)

def test_case_4():
    var_0 = 'echo'
    var_1 = '  line1  \n  line2  '
    var_2 = [var_0, var_1]
    var_3 = module_0.get_lines(var_2)