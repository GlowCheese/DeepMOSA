# Check out: https://github.com/GlowCheese/deepmosa
import builtins as module_3
import enum as module_1
import re as module_2

import pyrsistent._transformations as module_0
import pytest


def test_case_0():
    pass

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
    var_0 = []
    var_1 = lambda idx: idx % var_0 == var_0
    var_2 = module_0._get_keys_and_values(var_0, var_1)
    var_3 = module_0.transform(var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'not_a_callable'
    module_0.transform(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 'abc'
    var_1 = bool(var_0 == [(0, 'a'), (1, 'b'), (2, 'c')])
    module_0.transform(var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = None
    var_2 = module_0.discard(var_0, var_1)
    var_3 = '[i0u\n<{c|NCr\rwE~m'
    var_2.__call__(var_2, var_3, module=var_2)

def test_case_10():
    var_0 = 'abc'
    var_1 = module_0._items(var_0)
    var_2 = bool(var_1 == [(0, 'a'), (1, 'b'), (2, 'c')])
    assert var_2 is True

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = []
    var_1 = module_2.RegexFlag.LOCALE
    var_2 = module_0._get_keys_and_values(var_0, var_1)
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
    var_3 = b'\x03)\x11\xf3'
    module_0.transform(var_1, var_3)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = -2938
    var_1 = lambda idx: idx % var_0 == var_0
    module_0._get_keys_and_values(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = -2938
    var_1 = [var_0]
    var_2 = lambda idx: idx % var_0 == var_0
    module_0._get_keys_and_values(var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = 24
    var_1 = [var_0]
    var_2 = lambda k, v: v >= var_1
    module_0._get_keys_and_values(var_1, var_2)

def test_case_15():
    var_0 = []
    var_1 = lambda k, v: v >= var_0
    var_2 = module_0._get_keys_and_values(var_0, var_1)
    var_3 = module_0.ny(var_2)
    assert var_3 is True

def test_case_16():
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
    var_1 = lambda idx: idx % var_0 == var_0
    var_2 = module_1.EnumMeta
    with pytest.raises(ValueError):
        module_0._get_keys_and_values(var_1, var_2)

def test_case_17():
    var_0 = '^[a-z]+_[0-9]+$'
    var_1 = module_0.rex(var_0)
    var_2 = 'abc_123'
    var_3 = var_1(var_2)

def test_case_18():
    var_0 = '^test'
    var_1 = module_0.rex(var_0)
    var_2 = var_1(var_1)
    assert var_2 is False

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = []
    var_1 = lambda idx: idx % var_0 == var_0
    var_2 = module_0._get_keys_and_values(var_0, var_1)
    module_0.transform(var_2, var_1)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = []
    var_1 = lambda idx: idx % var_0 == var_0
    var_2 = var_1.__repr__()
    var_3 = [var_0, var_1, var_1]
    module_0.transform(var_2, var_3)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = []
    var_1 = lambda idx: idx % var_0 == var_0
    var_2 = module_0._get_keys_and_values(var_0, var_1)
    var_3 = var_1.__repr__()
    var_4 = [var_0, var_2, var_1]
    module_0.transform(var_3, var_4)

def test_case_22():
    var_0 = []
    var_1 = lambda idx: idx % var_0 == var_0
    var_2 = module_0._get_keys_and_values(var_0, var_1)
    var_3 = None
    var_4 = module_3.complex
    with pytest.raises(ValueError):
        module_0._get_keys_and_values(var_3, var_4)