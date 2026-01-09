# Check out: https://github.com/GlowCheese/deepmosa
import isort.hooks as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = True
    module_0.git_hook(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = module_0.git_hook()

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = True
    module_0.git_hook(directories=var_0)