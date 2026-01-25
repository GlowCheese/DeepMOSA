# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.wrap_modes as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.from_string(var_0)

def test_case_1():
    pass

def test_case_2():
    var_0 = ''
    var_1 = module_0.formatter_from_string(var_0)