# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyrsistent._transformations as module_0
import enum as module_1
import builtins as module_2

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

def test_case_6():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 3
    var_5 = {var_0: var_3, var_1: var_4, var_2: var_4}
    var_6 = module_0.discard(var_5, var_1)
    var_7 = {var_0: var_3, var_1: var_3, var_2: var_4}
    var_8 = 'd'
    var_9 = module_0.discard(var_7, var_8)
    var_10 = {}
    var_11 = module_0.discard(var_10, var_0)

def test_case_7():
    var_0 = None
    var_1 = module_0.ny(var_0)
    assert var_1 is True
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 2
    var_5 = 3
    var_6 = {var_1: var_4, var_2: var_4, var_3: var_5}
    var_7 = module_1._EnumDict()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'enum._EnumDict'
    assert len(var_7) == 0
    var_8 = module_0.transform(var_7, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'enum._EnumDict'
    assert len(var_8) == 0
    var_9 = module_0.discard(var_6, var_2)
    var_10 = {}
    var_11 = module_0.discard(var_10, var_9)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = module_2.float
    var_2 = [var_0, var_0, var_1, var_0]
    module_0.transform(var_0, var_2)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = '|kYjN'
    module_0.transform(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'dE'
    var_1 = None
    module_0.transform(var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    var_1 = module_2.float
    var_2 = [var_0, var_1]
    module_0.transform(var_0, var_2)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = '~IG@SA"|bqJ='
    var_1 = module_1._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    module_0.transform(var_1, var_0)