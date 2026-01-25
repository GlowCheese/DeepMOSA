# Check out: https://github.com/GlowCheese/deepmosa
import enum as module_1

import pyrsistent._transformations as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.transform(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.discard(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = '(HnH5-0Pba-jH%PGi'
    module_0.transform(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    module_0.inc(var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_0.dec(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    module_0.rex(var_0)

def test_case_6():
    var_0 = None
    var_1 = module_0.ny(var_0)
    assert var_1 is True

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = "(7\tSnH5-0Pba5j%'Gia"
    var_1 = None
    module_0.transform(var_1, var_0)

def test_case_8():
    var_0 = ()
    var_1 = module_0.transform(var_0, var_0)

def test_case_9():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = {var_0: var_3, var_1: var_4, var_2: var_4}
    var_6 = module_0.discard(var_5, var_0)
    var_7 = 'd'
    var_8 = module_0.discard(var_5, var_7)

def test_case_10():
    var_0 = None
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = module_0.transform(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = '^a'
    var_1 = module_1._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    module_0.transform(var_1, var_0)