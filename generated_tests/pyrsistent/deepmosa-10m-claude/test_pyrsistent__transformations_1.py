# Check out: https://github.com/GlowCheese/deepmosa
import enum as module_2
import inspect as module_3
import re as module_1

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
    var_0 = 'missing'
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
    var_0 = '^test'
    var_1 = module_1.purge()
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
    module_0.transform(var_1, var_0)

def test_case_8():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = lambda k: var_0
    var_2 = module_0._get_keys_and_values(var_0, var_1)
    var_3 = module_0.discard(var_0, var_1)

def test_case_9():
    var_0 = module_2.IntEnum
    var_1 = module_0._get_keys_and_values(var_0, var_0)
    var_2 = module_0.transform(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = '^0ent'
    var_1 = module_2._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
    module_0.transform(var_1, var_0)

def test_case_11():
    var_0 = '^\\d{3}-\\d{4}$'
    var_1 = module_0.rex(var_0)
    var_2 = var_1(var_0)

def test_case_12():
    var_0 = '^$'
    var_1 = module_0.rex(var_0)
    var_2 = var_1(var_1)

def test_case_13():
    var_0 = 'abc'
    var_1 = module_0._items(var_0)
    var_2 = bool(var_1 == [(0, 'a'), (1, 'b'), (2, 'c')])
    assert var_2 is True

def test_case_14():
    var_0 = module_2.IntEnum
    var_1 = module_0._get_keys_and_values(var_0, var_0)
    var_2 = module_0.ny(var_1)
    assert var_2 is True

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = module_1.RegexFlag
    var_1 = module_0._get_keys_and_values(var_0, var_0)
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
    module_0._get_keys_and_values(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = []
    var_1 = lambda idx, val: var_0
    var_2 = [var_0, var_1]
    module_0._get_keys_and_values(var_2, var_1)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = module_3.Parameter
    module_0._get_keys_and_values(var_0, var_0)

def test_case_18():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda : var_3
    with pytest.raises(ValueError):
        module_0._get_keys_and_values(var_2, var_4)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = []
    var_1 = lambda idx, val: var_0
    var_2 = module_0._get_keys_and_values(var_0, var_1)
    module_0.inc(var_0)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = lambda k: var_0
    var_2 = module_0._get_keys_and_values(var_0, var_1)
    var_3 = (var_2, var_2, var_2)
    module_0.transform(var_1, var_3)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = lambda x: var_5
    var_7 = []
    module_0._do_to_path(var_2, var_7, var_6)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = []
    var_1 = lambda idx, val: var_0
    var_2 = module_2._EnumDict
    module_0._get_keys_and_values(var_2, var_1)