# Check out: https://github.com/GlowCheese/deepmosa
import pyrsistent._immutable as module_0
import pytest


def test_case_0():
    var_0 = 'Sigl'
    var_1 = module_0.immutable(var_0, var_0)
    with pytest.raises(AttributeError):
        var_2 = var_1.value
    assert var_2 == 42

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
    var_0 = 'y'
    var_1 = [var_0, var_0]
    var_2 = 'Point'
    module_0.immutable(var_1, var_2)

def test_case_6():
    var_0 = 'x, y, id_'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)