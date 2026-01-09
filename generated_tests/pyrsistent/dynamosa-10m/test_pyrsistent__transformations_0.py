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
    var_0 = None
    module_0.dec(var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_0.rex(var_0)

def test_case_5():
    var_0 = None
    var_1 = module_0.ny(var_0)
    assert var_1 is True

def test_case_6():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_0.transform(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0

def test_case_7():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = None
    var_2 = module_0.discard(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = ')w\x0bL+*W\x0c[9avOmYkH'
    module_0.transform(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    var_1 = ')w\x0bL+*W\x0c[9avOmYkH'
    module_0.transform(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = ')w\x0bL+*W\x0c[9avOmYkH'
    module_0.transform(var_0, var_1)

def test_case_11():
    var_0 = None
    var_1 = (var_0, var_0)
    var_2 = module_0.transform(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = var_0.__reduce__()
    var_2 = (var_1, var_1)
    module_0.transform(var_1, var_2)