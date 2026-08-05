# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyrsistent._transformations as module_0
import enum as module_1
import re as module_2

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
    var_0 = '|[LgQp7L`)9j'
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
    var_0 = '`HN8(\nl4'
    var_1 = None
    module_0.transform(var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = module_1._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = var_1.__repr__()
    assert var_2 == '{}'
    var_3 = module_0.discard(var_1, var_0)
    var_4 = None
    var_5 = module_0.ny(var_4)
    assert var_5 is True
    module_2.escape(var_4)

def test_case_9():
    var_0 = ''
    var_1 = module_0.transform(var_0, var_0)
    assert var_1 == ''

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = ',\tvT'
    var_1 = module_1._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = 'MHN8]\nl4'
    var_3 = None
    var_4 = [var_3, var_1, var_0, var_2, var_1]
    module_0.transform(var_3, var_4)

def test_case_11():
    var_0 = '`HN8(\nl4'
    var_1 = None
    var_2 = [var_1, var_0]
    var_3 = module_0.transform(var_1, var_2)
    assert var_3 == '`HN8(\nl4'

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = 'q"|Ww|@+eF"v0Rp\x0br'
    var_1 = module_0.rex(var_0)
    var_2 = None
    var_3 = [var_2, var_1, var_0, var_1, var_1]
    module_0.transform(var_2, var_3)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = 'q"|Ww|@+eF"v0Rp\x0br'
    var_1 = module_0.rex(var_0)
    var_2 = 'mHN8]\nl4'
    var_3 = None
    var_4 = var_1.__hash__()
    var_5 = [var_3, var_1, var_0, var_2, var_1]
    module_0.transform(var_0, var_5)