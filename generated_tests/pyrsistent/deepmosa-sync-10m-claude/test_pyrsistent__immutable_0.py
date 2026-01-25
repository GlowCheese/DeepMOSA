# Check out: https://github.com/GlowCheese/deepmosa
import pyrsistent._immutable as module_0
import pytest


def test_case_0():
    var_0 = 'x, id_'
    var_1 = 'Data'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 'id1'
    var_5 = var_2(var_3, var_4)

def test_case_1():
    var_0 = module_0.immutable()

def test_case_2():
    var_0 = 'x, y, id_'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    with pytest.raises(SyntaxError):
        module_0.immutable(name=var_2, verbose=var_2)

def test_case_3():
    var_0 = module_0.immutable()
    var_1 = []
    var_2 = None
    with pytest.raises(SyntaxError):
        module_0.immutable(var_1, var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 'eRji+.-\rcd+\x0cc'
    var_1 = None
    module_0.immutable(var_0, verbose=var_1)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    module_0.immutable(var_0, var_0, var_0)