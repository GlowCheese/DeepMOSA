# Check out: https://github.com/GlowCheese/deepmosa
import pyrsistent._immutable as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = 'eRji+.-\rcd+\x0cc'
    var_1 = None
    module_0.immutable(var_0, verbose=var_1)

def test_case_1():
    var_0 = module_0.immutable()

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = module_0.immutable(verbose=var_0)
    var_2 = 'eRji+.-\rcd+\x0cc'
    module_0.immutable(var_2, verbose=var_1)

def test_case_3():
    var_0 = module_0.immutable()
    var_1 = []
    var_2 = None
    with pytest.raises(SyntaxError):
        module_0.immutable(var_1, var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_0.immutable(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    var_1 = module_0.immutable(verbose=var_0)
    var_2 = '\r&`VtCa$M8%OQl_'
    var_3 = None
    module_0.immutable(var_2, verbose=var_3)