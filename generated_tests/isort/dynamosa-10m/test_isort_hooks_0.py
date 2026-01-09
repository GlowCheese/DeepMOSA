# Check out: https://github.com/GlowCheese/deepmosa
import isort.hooks as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = True
    module_0.git_hook(modify=var_0, lazy=var_0, directories=var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    module_0.git_hook()