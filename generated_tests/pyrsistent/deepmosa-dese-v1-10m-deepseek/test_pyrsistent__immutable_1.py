# Check out: https://github.com/GlowCheese/deepmosa
import pyrsistent._immutable as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = ',v3tB`4m'
    var_1 = None
    module_0.immutable(var_0, verbose=var_1)

def test_case_1():
    var_0 = module_0.immutable()

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = module_0.immutable()
    var_1 = module_0.immutable(verbose=var_0)
    module_0.immutable(var_0)

def test_case_3():
    var_0 = None
    with pytest.raises(SyntaxError):
        module_0.immutable(name=var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_0.immutable(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = '_'
    var_1 = module_0.immutable()
    module_0.immutable(var_0, verbose=var_1)