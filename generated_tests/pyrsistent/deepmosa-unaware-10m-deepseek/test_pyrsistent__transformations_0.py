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
    var_0 = module_1.IntFlag
    var_1 = module_0.transform(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum.EnumMeta'
    assert len(var_1) == 0

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = module_2.RegexFlag
    module_0.transform(var_0, var_0)

def test_case_8():
    var_0 = module_2.purge()
    assert module_2.ASCII == module_2.RegexFlag.ASCII
    assert module_2.A == module_2.RegexFlag.ASCII
    assert module_2.IGNORECASE == module_2.RegexFlag.IGNORECASE
    assert module_2.I == module_2.RegexFlag.IGNORECASE
    assert module_2.LOCALE == module_2.RegexFlag.LOCALE
    assert module_2.L == module_2.RegexFlag.LOCALE
    assert module_2.UNICODE == module_2.RegexFlag.UNICODE
    assert module_2.U == module_2.RegexFlag.UNICODE
    assert module_2.MULTILINE == module_2.RegexFlag.MULTILINE
    assert module_2.M == module_2.RegexFlag.MULTILINE
    assert module_2.DOTALL == module_2.RegexFlag.DOTALL
    assert module_2.S == module_2.RegexFlag.DOTALL
    assert module_2.VERBOSE == module_2.RegexFlag.VERBOSE
    assert module_2.X == module_2.RegexFlag.VERBOSE
    assert module_2.TEMPLATE == module_2.RegexFlag.TEMPLATE
    assert module_2.T == module_2.RegexFlag.TEMPLATE
    assert module_2.DEBUG == module_2.RegexFlag.DEBUG
    var_1 = (var_0, var_0, var_0, var_0)
    var_2 = module_0.transform(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    var_1 = var_0.__str__()
    assert var_1 == 'None'
    module_0.transform(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = None
    var_2 = module_0.discard(var_0, var_1)
    var_3 = '[i0u\n<{c|NCr\rwE~m'
    var_2.__call__(var_2, var_3, module=var_2)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = module_2.purge()
    assert module_2.ASCII == module_2.RegexFlag.ASCII
    assert module_2.A == module_2.RegexFlag.ASCII
    assert module_2.IGNORECASE == module_2.RegexFlag.IGNORECASE
    assert module_2.I == module_2.RegexFlag.IGNORECASE
    assert module_2.LOCALE == module_2.RegexFlag.LOCALE
    assert module_2.L == module_2.RegexFlag.LOCALE
    assert module_2.UNICODE == module_2.RegexFlag.UNICODE
    assert module_2.U == module_2.RegexFlag.UNICODE
    assert module_2.MULTILINE == module_2.RegexFlag.MULTILINE
    assert module_2.M == module_2.RegexFlag.MULTILINE
    assert module_2.DOTALL == module_2.RegexFlag.DOTALL
    assert module_2.S == module_2.RegexFlag.DOTALL
    assert module_2.VERBOSE == module_2.RegexFlag.VERBOSE
    assert module_2.X == module_2.RegexFlag.VERBOSE
    assert module_2.TEMPLATE == module_2.RegexFlag.TEMPLATE
    assert module_2.T == module_2.RegexFlag.TEMPLATE
    assert module_2.DEBUG == module_2.RegexFlag.DEBUG
    var_1 = var_0.__repr__()
    assert var_1 == 'None'
    module_0.transform(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = var_0.__str__()
    assert var_1 == '{}'
    module_0.transform(var_0, var_1)