# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'Empty'
    var_1 = module_0.immutable(var_0, var_0)
    var_2 = repr(var_0)

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
    var_0 = 'T~W~JD'
    var_1 = 'Empty'
    module_0.immutable(var_0, var_1)

def test_case_6():
    var_0 = 'x, y, id_'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 17
    var_6 = var_2(var_3, var_4, id_=var_5)