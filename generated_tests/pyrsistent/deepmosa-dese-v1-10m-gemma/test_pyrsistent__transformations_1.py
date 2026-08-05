# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import pyrsistent._transformations as module_0
import enum as module_1
import re as module_2
import inspect as module_3

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
    var_0 = 'jo5D`bgd6$\x0byE8qg'
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
    var_0 = []
    var_1 = lambda k: k == var_0
    var_2 = module_0._get_keys_and_values(var_0, var_1)
    module_0._get_keys_and_values(var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = {}
    var_1 = None
    var_2 = module_0.discard(var_0, var_1)
    var_3 = module_1._EnumDict()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'enum._EnumDict'
    assert len(var_3) == 0
    var_4 = lambda k, v: v > var_3
    var_5 = module_0._get_keys_and_values(var_3, var_4)
    var_6 = var_4.__dir__()
    module_0._get_keys_and_values(var_6, var_4)

def test_case_9():
    var_0 = []
    var_1 = module_0.transform(var_0, var_0)

def test_case_10():
    var_0 = module_2.RegexFlag.MULTILINE
    var_1 = module_1._EnumDict()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'enum._EnumDict'
    assert len(var_1) == 0
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
    var_2 = lambda k, v: v > var_1
    var_3 = module_0._get_keys_and_values(var_1, var_2)
    var_4 = module_0._get_keys_and_values(var_3, var_0)

def test_case_11():
    var_0 = '\\d+'
    var_1 = module_0.rex(var_0)
    var_2 = module_0._get_keys_and_values(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = module_2.RegexFlag.UNICODE
    var_1 = var_0.__str__()
    assert var_1 == 're.UNICODE'
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
    var_2 = lambda k, v: v > var_1
    module_0._get_keys_and_values(var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = None
    var_1 = module_0.ny(var_0)
    assert var_1 is True
    var_2 = lambda k, v: v > var_1
    module_0._get_keys_and_values(var_2, var_2)

def test_case_14():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = lambda : var_3
    with pytest.raises(ValueError):
        module_0._get_keys_and_values(var_2, var_4)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = 1
    var_1 = lambda k: k == var_0
    var_2 = module_2.error
    module_0._get_keys_and_values(var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = []
    var_1 = None
    var_2 = [var_0, var_1, var_1, var_1]
    var_3 = module_0.transform(var_1, var_2)
    module_0._get_keys_and_values(var_0, var_3)

def test_case_17():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = lambda k, v: v > var_0
    var_2 = module_0._get_keys_and_values(var_0, var_1)
    var_3 = module_0.ny(var_0)
    assert var_3 is True

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = []
    var_1 = None
    var_2 = [var_0, var_1, var_1, var_1]
    var_3 = module_0.transform(var_1, var_2)
    var_4 = lambda k: k == var_0
    var_5 = module_1._EnumDict
    module_0._get_keys_and_values(var_5, var_4)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = lambda x: x[var_0]
    module_0._do_to_path(var_2, var_3, var_4)
    assert var_5 == 1

def test_case_20():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda x: x
    var_4 = module_0._get_keys_and_values(var_2, var_3)

def test_case_21():
    var_0 = module_1._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = lambda k, v: v > var_0
    var_2 = module_0._get_keys_and_values(var_0, var_1)
    var_3 = var_1.__dir__()
    var_4 = module_3.Parameter
    with pytest.raises(ValueError):
        module_0._get_keys_and_values(var_3, var_4)