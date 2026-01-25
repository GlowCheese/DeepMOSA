# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'Point'
    var_1 = module_0.immutable(var_0, var_0)
    var_2 = tuple()
    var_3 = 'x, y, id_'
    var_4 = module_0.immutable(var_3, var_0)

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
    var_0 = '!(\rK\t'
    var_1 = 'Point'
    var_2 = module_0.immutable()
    module_0.immutable(var_0, var_1)