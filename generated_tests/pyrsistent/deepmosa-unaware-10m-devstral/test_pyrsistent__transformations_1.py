# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyrsistent._transformations as module_0
import re as module_1
import enum as module_2

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
    var_0 = module_1.purge()
    assert module_1.ASCII == module_1.RegexFlag.ASCII
    assert module_1.A == module_1.RegexFlag.ASCII
    assert module_1.IGNORECASE == module_1.RegexFlag.IGNORECASE
    assert module_1.I == module_1.RegexFlag.IGNORECASE
    assert module_1.LOCALE == module_1.RegexFlag.LOCALE
    assert module_1.L == module_1.RegexFlag.LOCALE
    assert module_1.UNICODE == module_1.RegexFlag.UNICODE
    assert module_1.U == module_1.RegexFlag.UNICODE
    assert module_1.MULTILINE == module_1.RegexFlag.MULTILINE
    assert module_1.M == module_1.RegexFlag.MULTILINE
    assert module_1.DOTALL == module_1.RegexFlag.DOTALL
    assert module_1.S == module_1.RegexFlag.DOTALL
    assert module_1.VERBOSE == module_1.RegexFlag.VERBOSE
    assert module_1.X == module_1.RegexFlag.VERBOSE
    assert module_1.TEMPLATE == module_1.RegexFlag.TEMPLATE
    assert module_1.T == module_1.RegexFlag.TEMPLATE
    assert module_1.DEBUG == module_1.RegexFlag.DEBUG
    var_1 = var_0.__repr__()
    assert var_1 == 'None'
    module_0.transform(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = module_2._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    var_2 = var_1.__repr__()
    assert var_2 == '{}'
    var_3 = module_0.discard(var_1, var_0)
    var_4 = None
    var_5 = module_0.ny(var_4)
    assert var_5 is True
    module_1.escape(var_4)

def test_case_9():
    var_0 = ''
    var_1 = module_0.transform(var_0, var_0)
    assert var_1 == ''

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = var_0.__repr__()
    assert var_1 == '{}'
    module_0.transform(var_0, var_1)

def test_case_11():
    var_0 = None
    var_1 = [var_0, var_0]
    var_2 = module_0.transform(var_0, var_1)