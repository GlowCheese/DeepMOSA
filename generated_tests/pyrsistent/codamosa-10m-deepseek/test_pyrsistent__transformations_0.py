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
    var_0 = None
    module_0.inc(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = b'\xc9\x1b\xc1'
    module_0.dec(var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_0.rex(var_0)

def test_case_5():
    var_0 = None
    var_1 = module_0.ny(var_0)
    assert var_1 is True

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = None
    var_2 = module_0.discard(var_0, var_1)
    var_2.__delattr__(var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = b'\xc9\x1b\xc1\x9a'
    module_0.transform(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = "\t_V'1Wsg>vD11ciGN\x0cov"
    module_0.transform(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = "\t_V'1Wsg>vD1cGN\x0cov"
    var_1 = None
    module_0.transform(var_1, var_0)

def test_case_10():
    var_0 = None
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = module_0.transform(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = ''
    var_1 = module_1.auto
    var_2 = (-430.55462+948.80115j)
    var_3 = (var_0, var_1, var_0, var_2)
    module_0.transform(var_3, var_3)